//! GPU-only воркер апскейла 4x-UltraSharp.
//!
//! Инференс строго на CUDA: сессия создаётся с `session.disable_cpu_ep_fallback`,
//! поэтому ORT откажется коммитить граф, если хоть один узел не ляжет на GPU.
//! Картинки тайлятся с перекрытием и блендингом; видео идёт через ffmpeg-пайпы
//! (декод NVDEC → кадры → CUDA-инференс → NVENC), аудио копируется без перекодирования.

use std::fs;
use std::io::{BufRead, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use image::{DynamicImage, ImageBuffer, Rgb, Rgba};
use ort::session::Session;
use ort::value::Tensor;
use serde::Deserialize;
use serde_json::{json, Value};

// ------------------------------------------------------------------ конфигурация

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Config {
    source: String,
    output: Option<String>,
    model: String,
    #[serde(default)]
    tile: Option<u32>,
    #[serde(default)]
    overlap: Option<u32>,
    #[serde(default)]
    codec: Option<String>,
    #[serde(default)]
    quality: Option<String>,
    #[serde(default)]
    start_sec: Option<f64>,
    #[serde(default)]
    duration_sec: Option<f64>,
    #[serde(default)]
    fps: Option<f64>,
    #[serde(default)]
    ffmpeg_path: Option<String>,
    #[serde(default)]
    ffprobe_path: Option<String>,
    #[serde(default)]
    overwrite: Option<bool>,
}

impl Config {
    fn tile(&self) -> u32 {
        self.tile.unwrap_or(256).clamp(64, 512)
    }
    fn overlap(&self) -> u32 {
        let tile = self.tile();
        self.overlap.unwrap_or(16).min(tile / 4)
    }
    fn ffmpeg(&self) -> String {
        self.ffmpeg_path.clone().unwrap_or_else(|| "ffmpeg".to_string())
    }
    fn ffprobe(&self) -> String {
        self.ffprobe_path.clone().unwrap_or_else(|| "ffprobe".to_string())
    }
}

// ------------------------------------------------------------------ события JSONL

fn emit(value: Value) {
    let line = value.to_string();
    let stdout = std::io::stdout();
    let mut lock = stdout.lock();
    let _ = writeln!(lock, "{line}");
    let _ = lock.flush();
}

fn emit_log(job: &str, message: impl AsRef<str>) {
    emit(json!({"type": "log", "id": job, "message": message.as_ref()}));
}

fn emit_progress(job: &str, frame: usize, total: Option<usize>, started: Instant, phase: &str) {
    let elapsed = started.elapsed().as_secs_f64().max(0.001);
    let fps = frame as f64 / elapsed;
    let percent = total.map(|total| (frame as f64 / total.max(1) as f64) * 100.0);
    let eta_ms = match total {
        Some(total) if fps > 0.0 && frame > 0 => {
            let rest = total.saturating_sub(frame) as f64;
            Some((rest / fps * 1000.0).round() as u64)
        }
        _ => None,
    };
    emit(json!({
        "type": "progress",
        "id": job,
        "frame": frame,
        "totalFrames": total,
        "fps": (fps * 100.0).round() / 100.0,
        "percent": percent.map(|value| (value * 10.0).round() / 10.0),
        "etaMs": eta_ms,
        "phase": phase,
    }));
}

// ------------------------------------------------------------------ CUDA-сессия

struct Upscaler {
    session: Session,
    factor: u32,
}

impl Upscaler {
    /// Загружает модель строго на CUDA. Любая попытка уйти на CPU — ошибка.
    fn load(model: &Path) -> Result<Self> {
        if !model.is_file() {
            bail!("model not found: {}", model.display());
        }
        let env = ort::init().build().context("init onnxruntime")?;
        let builder = Session::builder(&env)?;
        let builder = builder
            .with_execution_providers([ort::ep::CUDA::default().with_device_id(0).build()])
            .map_err(|error| anyhow!("CUDA EP is unavailable: {error}"))?;
        let mut builder = builder
            .with_disable_cpu_fallback()
            .map_err(|error| anyhow!("cannot disable CPU fallback: {error}"))?;
        let session = builder.commit_from_file(model).map_err(|error| {
            anyhow!("GPU-only session failed (the model needs CPU ops): {error}")
        })?;
        let mut upscaler = Self { session, factor: 4 };
        // Проба: первый запуск инициализирует CUDA-ядра и даёт фактический масштаб.
        let probe = vec![0.25f32; 3 * 64 * 64];
        let output = upscaler.infer(64, &probe)?;
        if output.len() % (64 * 64 * 3) != 0 {
            bail!("unexpected model output size: {}", output.len());
        }
        // 64x64 -> out_w*out_h*3; при 4x это 3*256*256.
        let pixels = output.len() / 3;
        let side = (pixels as f64).sqrt().round() as u32;
        let factor = side / 64;
        if factor < 2 || factor > 8 {
            bail!("model scale factor {factor}x is not supported");
        }
        upscaler.factor = factor;
        Ok(upscaler)
    }

    fn infer(&mut self, tile: u32, input: &[f32]) -> Result<Vec<f32>> {
        let side = tile as usize;
        let tensor = Tensor::from_array((
            [1usize, 3, side, side],
            input.to_vec().into_boxed_slice(),
        ))?;
        let outputs = self.session.run(ort::inputs![tensor])?;
        let value = &outputs[0];
        let (_, data) = value.try_extract_tensor::<f32>()?;
        Ok(data.to_vec())
    }
}

// ------------------------------------------------------------------ тайлинг

fn tile_starts(len: usize, tile: usize, stride: usize) -> Vec<usize> {
    if len <= tile {
        return vec![0];
    }
    let mut starts = Vec::new();
    let mut start = 0usize;
    while start + tile < len {
        starts.push(start);
        start += stride.max(1);
    }
    starts.push(len - tile);
    starts.dedup();
    starts
}

/// Одномерный профиль веса вдоль оси: 1 в центре, линейный спад в перекрытии.
fn axis_weights(out_side: usize, factor: usize, overlap: usize, low: bool, high: bool) -> Vec<f32> {
    let mut weights = vec![1f32; out_side];
    let ramp = overlap * factor;
    if ramp > 0 {
        for position in 0..out_side {
            let mut weight = 1f32;
            if low && position < ramp {
                weight = weight.min((position as f32 + 0.5) / ramp as f32);
            }
            if high {
                let from_high = out_side - position;
                if from_high < ramp {
                    weight = weight.min((from_high as f32 - 0.5) / ramp as f32);
                }
            }
            weights[position] = weight.clamp(0.001, 1.0);
        }
    }
    weights
}

/// Апскейл RGB-кадра тайлами с перекрытием и блендингом.
fn upscale_rgb(
    upscaler: &mut Upscaler,
    rgb: &[u8],
    width: usize,
    height: usize,
    tile: u32,
    overlap: u32,
    mut on_tile: impl FnMut(usize, usize),
) -> Result<Vec<u8>> {
    let factor = upscaler.factor as usize;
    let tile = tile as usize;
    let overlap = overlap as usize;
    let stride = tile.saturating_sub(overlap).max(1);
    let starts_x = tile_starts(width, tile, stride);
    let starts_y = tile_starts(height, tile, stride);
    let out_w = width * factor;
    let out_h = height * factor;
    let total_tiles = starts_x.len() * starts_y.len();
    let mut accum = vec![0f32; out_w * out_h * 3];
    let mut weights = vec![0f32; out_w * out_h];
    let mut done = 0usize;

    for &start_y in &starts_y {
        for &start_x in &starts_x {
            let mut input = vec![0f32; 3 * tile * tile];
            for ty in 0..tile {
                let sy = (start_y + ty).min(height - 1);
                for tx in 0..tile {
                    let sx = (start_x + tx).min(width - 1);
                    let source = (sy * width + sx) * 3;
                    for channel in 0..3 {
                        input[(channel * tile + ty) * tile + tx] =
                            rgb[source + channel] as f32 / 255.0;
                    }
                }
            }
            let output = upscaler.infer(tile as u32, &input)?;
            if std::env::var("GPUTOOL_UPSCALE_DEBUG").is_ok() && done == 0 {
                let mut means = [0f32; 3];
                for channel in 0..3 {
                    let mut sum = 0f64;
                    for index in 0..tile * tile {
                        sum += input[(channel * tile * tile) + index] as f64;
                    }
                    means[channel] = (sum / (tile * tile) as f64) as f32;
                }
                let mut out_means = [0f32; 3];
                let out_side = tile * factor;
                for channel in 0..3 {
                    let mut sum = 0f64;
                    for index in 0..out_side * out_side {
                        sum += output[(channel * out_side * out_side) + index].clamp(0.0, 1.0) as f64;
                    }
                    out_means[channel] = (sum / (out_side * out_side) as f64) as f32;
                }
                eprintln!("debug input means {:?} output means {:?}", means, out_means);
            }
            let out_side = tile * factor;
            let left_overlap = start_x > 0;
            let right_overlap = start_x + tile < width;
            let top_overlap = start_y > 0;
            let bottom_overlap = start_y + tile < height;
            let weight_x = axis_weights(out_side, factor, overlap, left_overlap, right_overlap);
            let weight_y = axis_weights(out_side, factor, overlap, top_overlap, bottom_overlap);
            for oy in 0..out_side {
                let py = start_y * factor + oy;
                if py >= out_h {
                    continue;
                }
                for ox in 0..out_side {
                    let px = start_x * factor + ox;
                    if px >= out_w {
                        continue;
                    }
                    let mut weight = weight_x[ox].min(weight_y[oy]);
                    if weight <= 0.0 {
                        weight = 0.001;
                    }
                    let index = (py * out_w + px) * 3;
                    for channel in 0..3 {
                        // Выход модели — NCHW: канал это отдельная плоскость.
                        let tile_index = channel * out_side * out_side + oy * out_side + ox;
                        let value = output
                            .get(tile_index)
                            .copied()
                            .unwrap_or(0.0)
                            .clamp(0.0, 1.0);
                        accum[index + channel] += value * weight;
                    }
                    weights[py * out_w + px] += weight;
                }
            }
            done += 1;
            on_tile(done, total_tiles);
        }
    }

    let mut result = vec![0u8; out_w * out_h * 3];
    for pixel in 0..out_w * out_h {
        let weight = weights[pixel].max(0.0001);
        for channel in 0..3 {
            let value = accum[pixel * 3 + channel] / weight;
            result[pixel * 3 + channel] = (value.clamp(0.0, 1.0) * 255.0).round() as u8;
        }
    }
    Ok(result)
}

// ------------------------------------------------------------------ картинки

fn is_image(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|value| value.to_str()).unwrap_or("").to_ascii_lowercase().as_str(),
        "png" | "jpg" | "jpeg" | "webp" | "bmp"
    )
}

fn run_image(job: &str, config: &Config, upscaler: &mut Upscaler) -> Result<Value> {
    let source = Path::new(&config.source);
    let started = Instant::now();
    let image = image::open(source)
        .with_context(|| format!("cannot decode {}", source.display()))?;
    let has_alpha = matches!(image, DynamicImage::ImageRgba8(_) | DynamicImage::ImageLumaA8(_));
    let rgb_image = image.to_rgb8();
    let (width, height) = (rgb_image.width() as usize, rgb_image.height() as usize);
    let pixel_count = width * height;
    if pixel_count > 16_000_000 {
        bail!(
            "image is too large for tiled accumulation ({} MPix); split it first",
            pixel_count / 1_000_000
        );
    }
    emit_log(job, format!("image {width}x{height}, tile={} overlap={} alpha={has_alpha}", config.tile(), config.overlap()));
    let rgb = rgb_image.into_raw();
    let tile = config.tile();
    let overlap = config.overlap();
    let mut last_emit = Instant::now();
    let upscaled = upscale_rgb(upscaler, &rgb, width, height, tile, overlap, |done, total| {
        if last_emit.elapsed().as_millis() >= 300 || done == total {
            emit_progress(job, done, Some(total), started, "tiles");
            last_emit = Instant::now();
        }
    })?;
    let factor = upscaler.factor as usize;
    let out_w = (width * factor) as u32;
    let out_h = (height * factor) as u32;
    let output_path = resolve_output(config, source, "png");

    let buffer: ImageBuffer<Rgb<u8>, Vec<u8>> =
        ImageBuffer::from_raw(out_w, out_h, upscaled).ok_or_else(|| anyhow!("bad output buffer"))?;
    if has_alpha {
        let alpha = image.to_rgba8();
        let alpha = image::imageops::resize(&alpha, out_w, out_h, image::imageops::FilterType::Lanczos3);
        let mut rgba: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::from_raw(out_w, out_h, vec![0u8; out_w as usize * out_h as usize * 4])
            .ok_or_else(|| anyhow!("bad alpha buffer"))?;
        for y in 0..out_h as usize {
            for x in 0..out_w as usize {
                let pixel = buffer.get_pixel(x as u32, y as u32);
                let a = alpha.get_pixel(x as u32, y as u32)[3];
                rgba.put_pixel(x as u32, y as u32, Rgba([pixel[0], pixel[1], pixel[2], a]));
            }
        }
        DynamicImage::ImageRgba8(rgba).save(&output_path)?;
    } else {
        DynamicImage::ImageRgb8(buffer).save(&output_path)?;
    }
    let bytes = fs::metadata(&output_path).map(|meta| meta.len()).unwrap_or(0);
    Ok(json!({
        "engine": "rust",
        "device": "cuda:0",
        "providers": ["CUDAExecutionProvider"],
        "scale": factor,
        "input": {"width": width, "height": height},
        "output": output_path.to_string_lossy(),
        "width": out_w,
        "height": out_h,
        "bytes": bytes,
        "elapsedMs": started.elapsed().as_millis(),
    }))
}

fn resolve_output(config: &Config, source: &Path, extension: &str) -> PathBuf {
    if let Some(output) = &config.output {
        return PathBuf::from(output);
    }
    let stem = source.file_stem().and_then(|value| value.to_str()).unwrap_or("output");
    let dir = source.parent().unwrap_or_else(|| Path::new("."));
    dir.join(format!("{stem}.4x.{extension}"))
}

// ------------------------------------------------------------------ видео

#[derive(Debug, Deserialize)]
struct ProbeStream {
    codec_type: Option<String>,
    width: Option<u32>,
    height: Option<u32>,
    avg_frame_rate: Option<String>,
    duration: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ProbeFormat {
    duration: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Probe {
    streams: Vec<ProbeStream>,
    format: Option<ProbeFormat>,
}

fn probe_video(config: &Config, source: &Path) -> Result<(u32, u32, f64, f64, bool)> {
    let output = Command::new(config.ffprobe())
        .args(["-v", "error", "-print_format", "json", "-show_format", "-show_streams"])
        .arg(source)
        .output()
        .context("ffprobe failed to start")?;
    if !output.status.success() {
        bail!("ffprobe failed: {}", String::from_utf8_lossy(&output.stderr).trim());
    }
    let probe: Probe = serde_json::from_slice(&output.stdout).context("bad ffprobe json")?;
    let video = probe
        .streams
        .iter()
        .find(|stream| stream.codec_type.as_deref() == Some("video") && stream.width.unwrap_or(0) > 0)
        .ok_or_else(|| anyhow!("no video stream"))?;
    let width = video.width.unwrap_or(0);
    let height = video.height.unwrap_or(0);
    let mut fps = 0f64;
    if let Some((num, den)) = video.avg_frame_rate.as_deref().unwrap_or("").split_once('/') {
        let num: f64 = num.parse().unwrap_or(0.0);
        let den: f64 = den.parse().unwrap_or(0.0);
        if den > 0.0 {
            fps = num / den;
        }
    }
    if fps <= 0.0 {
        fps = 30.0;
    }
    let duration = probe
        .format
        .and_then(|format| format.duration)
        .or_else(|| video.duration.clone())
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(0.0);
    let has_audio = probe.streams.iter().any(|stream| stream.codec_type.as_deref() == Some("audio"));
    Ok((width, height, fps, duration, has_audio))
}

fn run_video(job: &str, config: &Config, upscaler: &mut Upscaler) -> Result<Value> {
    let source = Path::new(&config.source);
    let started = Instant::now();
    let (width, height, source_fps, source_duration, has_audio) = probe_video(config, source)?;
    let fps = config.fps.unwrap_or(source_fps);
    let start = config.start_sec.unwrap_or(0.0).max(0.0);
    let duration = config
        .duration_sec
        .unwrap_or_else(|| (source_duration - start).max(0.0));
    let total_frames = ((duration * fps).round() as usize).max(1);
    let factor = upscaler.factor as usize;
    let out_w = (width as usize * factor) as u32;
    let out_h = (height as usize * factor) as u32;
    let output_path = config
        .output
        .clone()
        .map(PathBuf::from)
        .unwrap_or_else(|| source.with_file_name(format!(
            "{}.4x.mp4",
            source.file_stem().and_then(|value| value.to_str()).unwrap_or("video")
        )));
    if output_path.exists() && config.overwrite != Some(true) {
        bail!("output already exists: {}", output_path.display());
    }
    let codec = config.codec.clone().unwrap_or_else(|| "hevc".to_string());
    let encoder = match codec.as_str() {
        "h264" => "h264_nvenc",
        _ => "hevc_nvenc",
    };
    let quality = config.quality.clone().unwrap_or_else(|| "balanced".to_string());
    let (preset, tune, rc) = match quality.as_str() {
        "fast" => ("p3", "ll", "27"),
        "quality" => ("p7", "hq", "19"),
        _ => ("p5", "hq", "22"),
    };

    emit_log(
        job,
        format!("video {width}x{height}@{source_fps:.2} -> {out_w}x{out_h}@{fps:.2}, tile={} overlap={}", config.tile(), config.overlap()),
    );

    let mut decoder_args: Vec<String> = vec![
        "-hide_banner".into(), "-nostdin".into(), "-loglevel".into(), "error".into(),
        "-hwaccel".into(), "cuda".into(),
    ];
    if start > 0.0 {
        decoder_args.extend(["-ss".into(), format!("{start:.3}")]);
    }
    if duration > 0.0 {
        decoder_args.extend(["-t".into(), format!("{duration:.3}")]);
    }
    decoder_args.extend([
        "-i".into(), source.to_string_lossy().into_owned(),
        "-map".into(), "0:v:0".into(),
        "-f".into(), "rawvideo".into(), "-pix_fmt".into(), "rgb24".into(), "-".into(),
    ]);
    let mut decoder = Command::new(config.ffmpeg())
        .args(&decoder_args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("cannot start ffmpeg decoder")?;

    let encoder_args: Vec<String> = vec![
        "-hide_banner".into(), "-nostdin".into(), "-loglevel".into(), "error".into(),
        "-f".into(), "rawvideo".into(), "-pix_fmt".into(), "rgb24".into(),
        "-s".into(), format!("{out_w}x{out_h}"), "-framerate".into(), format!("{fps}"),
        "-i".into(), "-".into(),
        "-i".into(), source.to_string_lossy().into_owned(),
        "-map".into(), "0:v:0".into(), "-map".into(), "1:a?".into(), "-c:a".into(), "copy".into(),
        "-c:v".into(), encoder.into(),
        "-gpu".into(), "0".into(),
        "-preset".into(), preset.into(), "-tune".into(), tune.into(), "-rc".into(), "vbr".into(), "-cq".into(), rc.into(),
        "-tag:v".into(), "hvc1".into(),
        "-movflags".into(), "+faststart".into(),
        "-y".into(), output_path.to_string_lossy().into_owned(),
    ];
    let mut encoder_process = Command::new(config.ffmpeg())
        .args(&encoder_args)
        .stdin(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("cannot start ffmpeg encoder")?;

    let mut decoder_out = decoder.stdout.take().context("decoder stdout")?;
    let mut encoder_in = encoder_process.stdin.take().context("encoder stdin")?;
    let frame_bytes = width as usize * height as usize * 3;
    let mut frame = vec![0u8; frame_bytes];
    let mut upscaled = vec![0u8; out_w as usize * out_h as usize * 3];
    let mut processed = 0usize;
    let mut last_emit = Instant::now();
    let tile = config.tile();
    let overlap = config.overlap();
    let mut failure: Option<anyhow::Error> = None;

    loop {
        match decoder_out.read_exact(&mut frame) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(error) => {
                failure = Some(anyhow!("decoder read failed: {error}"));
                break;
            }
        }
        match upscale_rgb(upscaler, &frame, width as usize, height as usize, tile, overlap, |_, _| {}) {
            Ok(next) => upscaled = next,
            Err(error) => {
                failure = Some(error);
                break;
            }
        }
        if let Err(error) = encoder_in.write_all(&upscaled) {
            failure = Some(anyhow!("encoder write failed: {error}"));
            break;
        }
        processed += 1;
        if last_emit.elapsed().as_millis() >= 500 || processed == total_frames {
            emit_progress(job, processed, Some(total_frames), started, "frames");
            last_emit = Instant::now();
        }
        if processed >= total_frames {
            break;
        }
    }
    drop(encoder_in);
    let decoder_status = decoder.wait().context("decoder wait")?;
    let encoder_status = encoder_process.wait().context("encoder wait")?;
    if let Some(error) = failure {
        let _ = fs::remove_file(&output_path);
        return Err(error);
    }
    if !decoder_status.success() {
        let _ = fs::remove_file(&output_path);
        let mut message = String::new();
        if let Some(mut stderr) = decoder.stderr.take() {
            let _ = stderr.read_to_string(&mut message);
        }
        bail!("decoder failed: {}", message.trim());
    }
    if !encoder_status.success() {
        let _ = fs::remove_file(&output_path);
        let mut message = String::new();
        if let Some(mut stderr) = encoder_process.stderr.take() {
            let _ = stderr.read_to_string(&mut message);
        }
        bail!("encoder failed: {}", message.trim());
    }
    let bytes = fs::metadata(&output_path).map(|meta| meta.len()).unwrap_or(0);
    Ok(json!({
        "engine": "rust",
        "device": "cuda:0",
        "providers": ["CUDAExecutionProvider"],
        "scale": factor,
        "frames": processed,
        "fps": fps,
        "hasAudio": has_audio,
        "output": output_path.to_string_lossy(),
        "width": out_w,
        "height": out_h,
        "bytes": bytes,
        "elapsedMs": started.elapsed().as_millis(),
    }))
}

// ------------------------------------------------------------------ прогон

fn run_job(job: &str, config: &Config, upscaler: &mut Option<Upscaler>) -> Result<Value> {
    if upscaler.is_none() {
        let started = Instant::now();
        let loaded = Upscaler::load(Path::new(&config.model))?;
        emit_log(job, format!("CUDA session ready in {} ms (scale {}x)", started.elapsed().as_millis(), loaded.factor));
        *upscaler = Some(loaded);
    }
    let engine = upscaler.as_mut().expect("upscaler");
    let source = Path::new(&config.source);
    if !source.is_file() {
        bail!("source not found: {}", source.display());
    }
    if is_image(source) {
        run_image(job, config, engine)
    } else {
        run_video(job, config, engine)
    }
}

fn probe(model: &Path, tile: u32) -> Result<()> {
    let started = Instant::now();
    let mut upscaler = Upscaler::load(model)?;
    let load_ms = started.elapsed().as_millis();
    let input = vec![0.3f32; 3 * tile as usize * tile as usize];
    upscaler.infer(tile, &input)?;
    let begin = Instant::now();
    upscaler.infer(tile, &input)?;
    let infer_ms = begin.elapsed().as_millis();
    emit(json!({
        "type": "probe",
        "ok": true,
        "device": "cuda:0",
        "providers": ["CUDAExecutionProvider"],
        "factor": upscaler.factor,
        "tile": tile,
        "loadMs": load_ms,
        "inferMs": infer_ms,
    }));
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let value = |flag: &str| -> Option<String> {
        args.iter().position(|item| item == flag).and_then(|index| args.get(index + 1)).cloned()
    };
    if args.iter().any(|item| item == "--probe") {
        let model = value("--model").ok_or_else(|| anyhow!("--model is required for --probe"))?;
        let tile = value("--tile").and_then(|item| item.parse().ok()).unwrap_or(256);
        return probe(Path::new(&model), tile);
    }
    if args.iter().any(|item| item == "--serve") {
        return serve();
    }
    let config_path = value("--config").ok_or_else(|| anyhow!("--config or --serve is required"))?;
    let text = fs::read_to_string(&config_path)?;
    let config: Config = serde_json::from_str(&text).context("bad config json")?;
    let mut upscaler: Option<Upscaler> = None;
    match run_job("once", &config, &mut upscaler) {
        Ok(result) => {
            emit(json!({"type": "result", "id": "once", "result": result}));
            Ok(())
        }
        Err(error) => {
            emit(json!({"type": "error", "id": "once", "code": classify(&error), "message": error.to_string()}));
            std::process::exit(1);
        }
    }
}

fn classify(error: &anyhow::Error) -> &'static str {
    let text = error.to_string();
    if text.contains("CUDA EP is unavailable") {
        "cuda-ep-unavailable"
    } else if text.contains("GPU-only session failed") {
        "gpu-unsupported"
    } else if text.contains("model not found") {
        "upscale-model-missing"
    } else if text.contains("source not found") {
        "input-missing"
    } else if text.contains("scale factor") {
        "upscale-factor-unsupported"
    } else {
        "upscale-failed"
    }
}

fn serve() -> Result<()> {
    let mut upscaler: Option<Upscaler> = None;
    emit(json!({"type": "ready", "engine": "rust" }));
    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let command: Value = match serde_json::from_str(&line) {
            Ok(value) => value,
            Err(error) => {
                emit(json!({"type": "error", "id": "", "code": "bad-command", "message": error.to_string()}));
                continue;
            }
        };
        let kind = command["type"].as_str().unwrap_or("");
        match kind {
            "ping" => emit(json!({"type": "pong"})),
            "shutdown" => break,
            "run" => {
                let job = command["id"].as_str().unwrap_or("job").to_string();
                let config: Config = match serde_json::from_value(command["config"].clone()) {
                    Ok(config) => config,
                    Err(error) => {
                        emit(json!({"type": "error", "id": job, "code": "bad-config", "message": error.to_string()}));
                        continue;
                    }
                };
                match run_job(&job, &config, &mut upscaler) {
                    Ok(result) => emit(json!({"type": "result", "id": job, "result": result})),
                    Err(error) => emit(json!({
                        "type": "error",
                        "id": job,
                        "code": classify(&error),
                        "message": error.to_string(),
                    })),
                }
            }
            other => emit(json!({"type": "error", "id": "", "code": "bad-command", "message": format!("unknown command {other}")})),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_starts_cover_the_image_without_gaps() {
        assert_eq!(tile_starts(100, 256, 240), vec![0]);
        assert_eq!(tile_starts(512, 256, 240), vec![0, 240, 256]);
        let starts = tile_starts(1000, 256, 240);
        assert_eq!(*starts.last().unwrap(), 1000 - 256);
        for pair in starts.windows(2) {
            assert!(pair[1] <= pair[0] + 240);
            assert!(pair[1] > pair[0]);
        }
    }

    #[test]
    fn weights_ramp_on_shared_edges_and_stay_one_inside() {
        let weights = axis_weights(8, 4, 1, true, true);
        assert!(weights[0] < 0.2);
        assert!((weights[4] - 1.0).abs() < f32::EPSILON);
        assert!(weights[7] < 0.2);
    }
}
