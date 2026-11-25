//! GPU-only x2-апскейл: SPAN fp16, батч кадров, без тайлинга.
//!
//! Инференс строго на CUDA (`disable_cpu_ep_fallback`), модель обязана быть
//! fp16 и давать ровно x2. Видео: NVDEC-декод → батч → NVENC-энкод (аудио copy).
//! Папка PNG: декод/энкод через image (CPU неизбежен для PNG), но батч держит GPU
//! загруженным; раздельные метрики decode/infer/encode попадают в результат.

use std::fs::{self, File};
use std::io::{BufRead, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use half::f16;
use image::codecs::png::{CompressionType, FilterType, PngEncoder};
use image::{ExtendedColorType, ImageEncoder};
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
    batch: Option<usize>,
    #[serde(default)]
    tile: Option<u32>,
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
    /// auto | cuda | tensorrt: провайдер инференса.
    #[serde(default)]
    provider: Option<String>,
    /// Каталог кэша TensorRT engine.
    #[serde(default)]
    engine_cache: Option<String>,
    /// Целевая ширина результата (должна быть чётной; модель всегда x2).
    #[serde(default)]
    width: Option<u32>,
    /// Целевая высота результата.
    #[serde(default)]
    height: Option<u32>,
}

impl Config {
    fn batch(&self) -> usize {
        self.batch.unwrap_or(8).clamp(1, 64)
    }
    fn ffmpeg(&self) -> String {
        self.ffmpeg_path.clone().unwrap_or_else(|| "ffmpeg".to_string())
    }
    fn ffprobe(&self) -> String {
        self.ffprobe_path.clone().unwrap_or_else(|| "ffprobe".to_string())
    }
    /// Полуразмер входа для целевого разрешения: модель x2 даёт ровно target.
    fn target_half(&self) -> Result<Option<(u32, u32)>> {
        match (self.width, self.height) {
            (None, None) => Ok(None),
            (Some(width), Some(height)) => {
                if width < 2 || height < 2 || width % 2 != 0 || height % 2 != 0 {
                    bail!("invalid-target: width/height должны быть чётными и >= 2");
                }
                Ok(Some((width / 2, height / 2)))
            }
            _ => bail!("invalid-target: width и height задаются только вместе"),
        }
    }

    fn check_tile(&self) -> Result<()> {
        if self.tile.unwrap_or(0) > 0 {
            bail!("tiling-not-supported: тайлинг удалён, кадр идёт целиком (tile=0)");
        }
        Ok(())
    }
}

// ------------------------------------------------------------------ события JSONL

fn emit(value: Value) {
    let stdout = std::io::stdout();
    let mut lock = stdout.lock();
    let _ = writeln!(lock, "{value}");
    let _ = lock.flush();
}

fn emit_log(job: &str, message: impl AsRef<str>) {
    emit(json!({"type": "log", "id": job, "message": message.as_ref()}));
}

fn emit_progress(
    job: &str,
    frame: usize,
    total: Option<usize>,
    started: Instant,
    phase: &str,
    fps_override: Option<f64>,
) {
    let elapsed = started.elapsed().as_secs_f64().max(0.001);
    let fps = fps_override.unwrap_or(frame as f64 / elapsed);
    let percent = total.map(|total| ((frame as f64 / total.max(1) as f64) * 100.0).min(100.0));
    let eta_ms = match total {
        Some(total) if fps > 0.0 && frame > 0 => Some(((total.saturating_sub(frame)) as f64 / fps * 1000.0).round() as u64),
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

/// Дренаж stderr ffmpeg в отдельном потоке.
///
/// Без него пайп stderr (64 КБ) заполняется и декодер/энкодер блокируется на
/// записи; воркер в это время ждёт `wait()` и задача «висит» навсегда.
fn drain_stderr(mut pipe: impl Read + Send + 'static) -> std::thread::JoinHandle<String> {
    std::thread::spawn(move || {
        let mut buffer = Vec::new();
        let _ = pipe.read_to_end(&mut buffer);
        String::from_utf8_lossy(&buffer).into_owned()
    })
}

// ------------------------------------------------------------------ CUDA-сессия

struct Upscaler {
    session: Session,
    provider: String,
}

impl Upscaler {
    fn load(model: &Path, provider: Option<&str>, engine_cache: Option<String>) -> Result<Self> {
        if !model.is_file() {
            bail!("model not found: {}", model.display());
        }
        let env = ort::init().build().context("init onnxruntime")?;
        let wanted = provider.unwrap_or("auto");
        let attempts: Vec<&str> = match wanted {
            "tensorrt" => vec!["tensorrt"],
            "cuda" => vec!["cuda"],
            _ => vec!["tensorrt", "cuda"],
        };
        let mut last_error: Option<String> = None;
        let mut built: Option<(Session, String)> = None;
        for provider in attempts {
            let result: Result<(Session, String)> = (|| {
                let raw = match provider {
                    "tensorrt" => {
                        let cache = engine_cache
                            .clone()
                            .or_else(|| std::env::var("GPUTOOL_TRT_CACHE").ok())
                            .unwrap_or_else(|| {
                                Path::new(model)
                                    .parent()
                                    .unwrap_or_else(|| Path::new("."))
                                    .join("trt-cache")
                                    .to_string_lossy()
                                    .into_owned()
                            });
                        let _ = fs::create_dir_all(&cache);
                        let builder = Session::builder(&env)?;
                        let builder = builder
                            .with_execution_providers([ort::ep::TensorRT::default()
                                .with_device_id(0)
                                .with_fp16(true)
                                .with_max_workspace_size(2 << 30)
                                .with_engine_cache(true)
                                .with_engine_cache_path(cache)
                                .with_timing_cache(true)
                                .build()])
                            .map_err(|error| anyhow!("tensorrt-unavailable: {error}"))?;
                        let mut builder = builder
                            .with_disable_cpu_fallback()
                            .map_err(|error| anyhow!("tensorrt-unavailable: {error}"))?;
                        let session = builder
                            .commit_from_file(model)
                            .map_err(|error| anyhow!("tensorrt-unavailable: {error}"))?;
                        (session, "TensorrtExecutionProvider".to_string())
                    }
                    _ => {
                        let builder = Session::builder(&env)?;
                        let builder = builder
                            .with_execution_providers([ort::ep::CUDA::default().with_device_id(0).build()])
                            .map_err(|error| anyhow!("cuda-unavailable: {error}"))?;
                        let mut builder = builder
                            .with_disable_cpu_fallback()
                            .map_err(|error| anyhow!("cuda-unavailable: {error}"))?;
                        let session = builder
                            .commit_from_file(model)
                            .map_err(|error| anyhow!("cuda-unavailable: {error}"))?;
                        (session, "CUDAExecutionProvider".to_string())
                    }
                };
                Ok(raw)
            })();
            match result {
                Ok(value) => {
                    built = Some(value);
                    break;
                }
                Err(error) => {
                    last_error = Some(error.to_string());
                    if wanted == "tensorrt" {
                        return Err(error);
                    }
                }
            }
        }
        let (session, provider) = built.ok_or_else(|| anyhow!("{}", last_error.unwrap_or_else(|| "no execution provider".to_string())))?;
        let mut upscaler = Self { session, provider };
        let probe = vec![f16::ZERO; 3 * 64 * 64];
        let output = upscaler.infer_raw(1, 64, 64, probe.as_slice()).map_err(|error| {
            anyhow!("model-fp16-required: не удалось выполнить fp16-инференс ({error}); нужна ONNX-модель с float16 I/O")
        })?;
        let pixels = output.len() / 3;
        let side = (pixels as f64).sqrt().round() as usize;
        if side == 0 || side % 64 != 0 || side / 64 != 2 {
            bail!("unsupported-scale: ожидается ровно x2, получено {}x", side as f64 / 64.0);
        }
        Ok(upscaler)
    }

    fn infer_raw(&mut self, batch: usize, width: usize, height: usize, input: &[f16]) -> Result<Vec<f16>> {
        let tensor = Tensor::from_array((
            [batch, 3, height, width],
            input.to_vec().into_boxed_slice(),
        ))?;
        let outputs = self.session.run(ort::inputs![tensor])?;
        let value = &outputs[0];
        let (_, data) = value.try_extract_tensor::<f16>()?;
        Ok(data.to_vec())
    }

    /// Батч RGB-кадров (u8) → батч x2 RGB-кадров (u8).
    fn upscale_batch(&mut self, frames: &[&[u8]], width: usize, height: usize) -> Result<Vec<Vec<u8>>> {
        let batch = frames.len();
        if batch == 0 {
            return Ok(Vec::new());
        }
        let plane = width * height;
        // Таблица u8/255 -> f16: без делений и float-конверсий на каждый пиксель.
        let lut: Vec<f16> = (0..=255u16).map(|value| f16::from_f32(value as f32 / 255.0)).collect();
        let mut input = vec![f16::ZERO; batch * 3 * plane];
        for (index, frame) in frames.iter().enumerate() {
            if frame.len() != plane * 3 {
                bail!("bad frame size {} (expected {})", frame.len(), plane * 3);
            }
            for channel in 0..3 {
                let base = (index * 3 + channel) * plane;
                for pixel in 0..plane {
                    input[base + pixel] = lut[frame[pixel * 3 + channel] as usize];
                }
            }
        }
        let output = self.infer_raw(batch, width, height, input.as_slice())?;
        let out_w = width * 2;
        let out_h = height * 2;
        let out_plane = out_w * out_h;
        if output.len() != batch * 3 * out_plane {
            bail!("unexpected model output {} values (batch={batch}, {out_w}x{out_h})", output.len());
        }
        let mut result = Vec::with_capacity(batch);
        for index in 0..batch {
            let mut frame = vec![0u8; out_plane * 3];
            for channel in 0..3 {
                let base = (index * 3 + channel) * out_plane;
                for pixel in 0..out_plane {
                    let value = output[base + pixel].to_f32().clamp(0.0, 1.0);
                    frame[pixel * 3 + channel] = (value * 255.0).round() as u8;
                }
            }
            result.push(frame);
        }
        Ok(result)
    }
}

// ------------------------------------------------------------------ IO кадров

fn is_image(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|value| value.to_str()).map(|value| value.to_ascii_lowercase()).as_deref(),
        Some("png" | "jpg" | "jpeg" | "webp" | "bmp")
    )
}

fn decode_frame(path: &Path) -> Result<(usize, usize, Vec<u8>)> {
    let image = image::open(path).with_context(|| format!("cannot decode {}", path.display()))?;
    let rgb = image.to_rgb8();
    Ok((rgb.width() as usize, rgb.height() as usize, rgb.into_raw()))
}

fn encode_frame(path: &Path, width: usize, height: usize, rgb: &[u8]) -> Result<()> {
    let file = BufWriter::new(File::create(path)?);
    let encoder = PngEncoder::new_with_quality(file, CompressionType::Fast, FilterType::Adaptive);
    encoder
        .write_image(rgb, width as u32, height as u32, ExtendedColorType::Rgb8)
        .with_context(|| format!("cannot encode {}", path.display()))?;
    Ok(())
}

fn collect_frames(source: &Path) -> Result<Vec<PathBuf>> {
    let mut frames: Vec<PathBuf> = if source.is_dir() {
        fs::read_dir(source)?
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| path.is_file() && is_image(path))
            .collect()
    } else {
        vec![source.to_path_buf()]
    };
    frames.sort();
    if frames.is_empty() {
        bail!("decode-failed: нет кадров в {}", source.display());
    }
    Ok(frames)
}

// ------------------------------------------------------------------ папка кадров / картинка

fn run_frames(job: &str, config: &Config, upscaler: &mut Upscaler) -> Result<Value> {
    let source = Path::new(&config.source);
    let frames = collect_frames(source)?;
    let single = source.is_file();
    let output_dir = if single {
        config
            .output
            .clone()
            .map(PathBuf::from)
            .unwrap_or_else(|| source.with_file_name(format!(
                "{}_x2.png",
                source.file_stem().and_then(|value| value.to_str()).unwrap_or("image")
            )))
    } else {
        config
            .output
            .clone()
            .map(PathBuf::from)
            .unwrap_or_else(|| source.to_path_buf())
    };
    if single {
        if let Some(parent) = output_dir.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
    } else {
        fs::create_dir_all(&output_dir)?;
    }
    let total = frames.len();
    let batch_size = config.batch();
    let started = Instant::now();
    let target = config.target_half()?;
    let mut decode_ns = 0u128;
    let mut infer_ns = 0u128;
    let mut encode_ns = 0u128;
    let mut done = 0usize;
    let mut last_emit = Instant::now();

    for chunk in frames.chunks(batch_size) {
        let mut decoded: Vec<(usize, usize, Vec<u8>)> = Vec::with_capacity(chunk.len());
        let t_decode = Instant::now();
        for path in chunk {
            let (width, height, data) = decode_frame(path)?;
            let (data, width, height) = prepare_frame(&data, width, height, target)?;
            decoded.push((width, height, data));
        }
        decode_ns += t_decode.elapsed().as_nanos();
        let (width, height) = (decoded[0].0, decoded[0].1);
        if decoded.iter().any(|(w, h, _)| *w != width || *h != height) {
            // Разные размеры — обрабатываем по одному, чтобы не ломать батч.
            for (index, path) in chunk.iter().enumerate() {
                let frame_data = decoded[index].2.clone();
                let t_infer = Instant::now();
                let mut out = upscaler.upscale_batch(&[frame_data.as_slice()], width, height)?;
                infer_ns += t_infer.elapsed().as_nanos();
                let t_encode = Instant::now();
                let target_path = output_path(single, path, &output_dir);
                encode_frame(&target_path, width * 2, height * 2, &out.remove(0))?;
                encode_ns += t_encode.elapsed().as_nanos();
                done += 1;
            }
            continue;
        }
        let refs: Vec<&[u8]> = decoded.iter().map(|(_, _, data)| data.as_slice()).collect();
        let t_infer = Instant::now();
        let outputs = upscaler.upscale_batch(&refs, width, height)?;
        infer_ns += t_infer.elapsed().as_nanos();
        let t_encode = Instant::now();
        for (index, path) in chunk.iter().enumerate() {
            let target = output_path(single, path, &output_dir);
            encode_frame(&target, width * 2, height * 2, &outputs[index])?;
        }
        encode_ns += t_encode.elapsed().as_nanos();
        done += chunk.len();
        if last_emit.elapsed().as_millis() >= 500 || done == total {
            emit_progress(job, done, Some(total), started, "frames", None);
            last_emit = Instant::now();
        }
    }
    let elapsed = started.elapsed();
    let fps = done as f64 / elapsed.as_secs_f64().max(0.001);
    Ok(json!({
        "engine": "rust",
        "device": "cuda:0",
        "providers": [upscaler.provider.clone()],
        "scale": 2,
        "batch": batch_size,
        "frames": done,
        "fps": (fps * 100.0).round() / 100.0,
        "inferMsPerFrame": (infer_ns as f64 / 1e6 / done.max(1) as f64 * 100.0).round() / 100.0,
        "decodeMsPerFrame": (decode_ns as f64 / 1e6 / done.max(1) as f64 * 100.0).round() / 100.0,
        "encodeMsPerFrame": (encode_ns as f64 / 1e6 / done.max(1) as f64 * 100.0).round() / 100.0,
        "elapsedMs": elapsed.as_millis(),
        "output": output_dir.to_string_lossy(),
        "directory": !single,
    }))
}

fn output_path(single: bool, source: &Path, output: &Path) -> PathBuf {
    if single {
        return output.to_path_buf();
    }
    let stem = source.file_stem().and_then(|value| value.to_str()).unwrap_or("frame");
    output.join(format!("{stem}_x2.png"))
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
        .ok_or_else(|| anyhow!("decode-failed: no video stream"))?;
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
    Ok((video.width.unwrap_or(0), video.height.unwrap_or(0), fps, duration, has_audio))
}

fn run_video(job: &str, config: &Config, upscaler: &mut Upscaler) -> Result<Value> {
    let source = Path::new(&config.source);
    let started = Instant::now();
    let (width, height, source_fps, source_duration, has_audio) = probe_video(config, source)?;
    let target = config.target_half()?;
    let fps = config.fps.unwrap_or(source_fps);
    let start = config.start_sec.unwrap_or(0.0).max(0.0);
    let duration = config.duration_sec.unwrap_or_else(|| (source_duration - start).max(0.0));
    let total_frames = ((duration * fps).round() as usize).max(1);
    let (in_w, in_h) = target.map(|(w, h)| (w, h)).unwrap_or((width, height));
    let out_w = in_w * 2;
    let out_h = in_h * 2;
    let output_path = config.output.clone().map(PathBuf::from).unwrap_or_else(|| {
        source.with_file_name(format!(
            "{}.x2.mp4",
            source.file_stem().and_then(|value| value.to_str()).unwrap_or("video")
        ))
    });
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
        format!("video {width}x{height}@{source_fps:.2} -> {out_w}x{out_h}@{fps:.2}, batch={}, no tiling", config.batch()),
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
        .context("decode-failed: cannot start ffmpeg")?;

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
        .context("encode-failed: cannot start ffmpeg")?;

    let mut decoder_out = decoder.stdout.take().context("decoder stdout")?;
    let mut encoder_in = encoder_process.stdin.take().context("encoder stdin")?;
    let decoder_err = decoder.stderr.take().map(drain_stderr);
    let encoder_err = encoder_process.stderr.take().map(drain_stderr);
    let frame_bytes = width as usize * height as usize * 3;
    let batch_size = config.batch();
    let mut processed = 0usize;
    let mut infer_ns = 0u128;
    let mut last_emit = Instant::now();
    let mut failure: Option<anyhow::Error> = None;

    'outer: loop {
        let mut frames: Vec<Vec<u8>> = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            let mut frame = vec![0u8; frame_bytes];
            match decoder_out.read_exact(&mut frame) {
                Ok(()) => {
                    let (data, _, _) = prepare_frame(&frame, width as usize, height as usize, target)?;
                    frames.push(data);
                }
                Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(error) => {
                    failure = Some(anyhow!("decode-failed: {error}"));
                    break 'outer;
                }
            }
        }
        if frames.is_empty() {
            break;
        }
        let refs: Vec<&[u8]> = frames.iter().map(|frame| frame.as_slice()).collect();
        let t_infer = Instant::now();
        let outputs = match upscaler.upscale_batch(&refs, in_w as usize, in_h as usize) {
            Ok(outputs) => outputs,
            Err(error) => {
                failure = Some(error);
                break;
            }
        };
        infer_ns += t_infer.elapsed().as_nanos();
        for output in outputs {
            if let Err(error) = encoder_in.write_all(&output) {
                failure = Some(anyhow!("encode-failed: {error}"));
                break 'outer;
            }
            processed += 1;
        }
        if last_emit.elapsed().as_millis() >= 500 {
            emit_progress(job, processed.min(total_frames), Some(total_frames), started, "frames", None);
            last_emit = Instant::now();
        }
    }
    // Счётчик кадров нельзя использовать как условие остановки: декодер может
    // отдать на пару кадров больше расчётного total_frames, и тогда читатель
    // закрывает пайп, а ffmpeg навсегда блокируется на записи в него.
    drop(encoder_in);
    drop(decoder_out);
    // На границе `-t` кадров может быть больше расчётного total_frames: в
    // финализации показываем фактическое число, иначе прогресс «застревает».
    emit_progress(job, processed, Some(total_frames.max(processed)), started, "finalize", None);
    let decoder_status = decoder.wait().context("decoder wait")?;
    let encoder_status = encoder_process.wait().context("encoder wait")?;
    let decoder_log = decoder_err.map(|handle| handle.join().unwrap_or_default()).unwrap_or_default();
    let encoder_log = encoder_err.map(|handle| handle.join().unwrap_or_default()).unwrap_or_default();
    if let Some(error) = failure {
        let _ = fs::remove_file(&output_path);
        return Err(error);
    }
    if !decoder_status.success() {
        let _ = fs::remove_file(&output_path);
        bail!("decode-failed: {}", decoder_log.trim());
    }
    if !encoder_status.success() {
        let _ = fs::remove_file(&output_path);
        bail!("encode-failed: {}", encoder_log.trim());
    }
    let bytes = fs::metadata(&output_path).map(|meta| meta.len()).unwrap_or(0);
    let elapsed = started.elapsed();
    Ok(json!({
        "engine": "rust",
        "device": "cuda:0",
        "providers": [upscaler.provider.clone()],
        "scale": 2,
        "batch": batch_size,
        "frames": processed,
        "fps": fps,
        "processedFps": ((processed as f64 / elapsed.as_secs_f64().max(0.001)) * 100.0).round() / 100.0,
        "inferMsPerFrame": (infer_ns as f64 / 1e6 / processed.max(1) as f64 * 100.0).round() / 100.0,
        "hasAudio": has_audio,
        "output": output_path.to_string_lossy(),
        "width": out_w,
        "height": out_h,
        "bytes": bytes,
        "elapsedMs": elapsed.as_millis(),
    }))
}

// ------------------------------------------------------------------ прогон

fn resize_rgb(data: &[u8], width: usize, height: usize, target_w: usize, target_h: usize) -> Result<Vec<u8>> {
    let image = image::RgbImage::from_raw(width as u32, height as u32, data.to_vec())
        .ok_or_else(|| anyhow!("bad frame buffer"))?;
    let resized = image::imageops::resize(&image, target_w as u32, target_h as u32, image::imageops::FilterType::Lanczos3);
    Ok(resized.into_raw())
}

fn prepare_frame(data: &[u8], width: usize, height: usize, target: Option<(u32, u32)>) -> Result<(Vec<u8>, usize, usize)> {
    match target {
        None => Ok((data.to_vec(), width, height)),
        Some((half_w, half_h)) if half_w as usize == width && half_h as usize == height => Ok((data.to_vec(), width, height)),
        Some((half_w, half_h)) => Ok((resize_rgb(data, width, height, half_w as usize, half_h as usize)?, half_w as usize, half_h as usize)),
    }
}

fn run_job(job: &str, config: &Config, upscaler: &mut Option<Upscaler>) -> Result<Value> {
    config.check_tile()?;
    config.target_half()?;
    if upscaler.is_none() {
        let started = Instant::now();
        let loaded = Upscaler::load(Path::new(&config.model), config.provider.as_deref(), config.engine_cache.clone())?;
        emit_log(job, format!("{} fp16 session ready in {} ms (x2)", loaded.provider, started.elapsed().as_millis()));
        *upscaler = Some(loaded);
    }
    let engine = upscaler.as_mut().expect("upscaler");
    let source = Path::new(&config.source);
    if !source.exists() {
        bail!("input-missing: {}", source.display());
    }
    if source.is_dir() || is_image(source) {
        run_frames(job, config, engine)
    } else {
        run_video(job, config, engine)
    }
}

fn probe(model: &Path) -> Result<()> {
    let started = Instant::now();
    let mut upscaler = Upscaler::load(model, None, None)?;
    let load_ms = started.elapsed().as_millis();
    let input = vec![f16::ZERO; 3 * 64 * 64];
    upscaler.infer_raw(1, 64, 64, input.as_slice())?;
    let begin = Instant::now();
    for _ in 0..3 {
        upscaler.infer_raw(1, 64, 64, input.as_slice())?;
    }
    let ms = begin.elapsed().as_secs_f64() * 1000.0 / 3.0;
    emit(json!({
        "type": "probe",
        "engine": "rust",
        "device": "cuda:0",
        "providers": [upscaler.provider.clone()],
        "scale": 2,
        "precision": "fp16",
        "loadMs": load_ms,
        "msPerFrame64": (ms * 100.0).round() / 100.0,
    }));
    Ok(())
}

fn classify(error: &anyhow::Error) -> &'static str {
    let text = error.to_string();
    if text.starts_with("cuda-unavailable") || text.contains("CUDA") {
        "cuda-unavailable"
    } else if text.starts_with("model-fp16-required") {
        "model-fp16-required"
    } else if text.starts_with("unsupported-scale") {
        "unsupported-scale"
    } else if text.starts_with("tiling-not-supported") {
        "tiling-not-supported"
    } else if text.starts_with("input-missing") {
        "input-missing"
    } else if text.starts_with("decode-failed") {
        "decode-failed"
    } else if text.starts_with("encode-failed") {
        "encode-failed"
    } else if text.starts_with("model not found") {
        "model-missing"
    } else {
        "upscale-failed"
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let value = |flag: &str| -> Option<String> {
        args.iter().position(|item| item == flag).and_then(|index| args.get(index + 1)).cloned()
    };
    if args.iter().any(|item| item == "--serve") {
        return serve();
    }
    if let Some(model) = value("--probe") {
        return probe(Path::new(&model));
    }
    let config_path = value("--config").ok_or_else(|| anyhow!("--config, --probe или --serve обязательны"))?;
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

fn serve() -> Result<()> {
    let mut upscaler: Option<Upscaler> = None;
    emit(json!({"type": "ready", "engine": "rust"}));
    for line in std::io::stdin().lock().lines() {
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
        match command["type"].as_str().unwrap_or("") {
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

    fn config(source: &str) -> Config {
        Config {
            provider: None,
            engine_cache: None,
            source: source.into(),
            output: None,
            model: "/tmp/m.onnx".into(),
            batch: None,
            tile: None,
            codec: None,
            quality: None,
            start_sec: None,
            duration_sec: None,
            fps: None,
            ffmpeg_path: None,
            ffprobe_path: None,
            overwrite: None,
            width: None,
            height: None,
        }
    }

    #[test]
    fn tile_is_rejected() {
        let mut cfg = config("/tmp/x");
        cfg.tile = Some(256);
        assert!(cfg.check_tile().is_err());
        cfg.tile = Some(0);
        assert!(cfg.check_tile().is_ok());
    }

    #[test]
    fn batch_defaults_and_clamps() {
        let mut cfg = config("/tmp/x");
        assert_eq!(cfg.batch(), 8);
        cfg.batch = Some(0);
        assert_eq!(cfg.batch(), 1);
        cfg.batch = Some(1000);
        assert_eq!(cfg.batch(), 64);
    }

    #[test]
    fn target_size_is_even_and_complete() {
        let mut cfg = config("/tmp/x");
        assert_eq!(cfg.target_half().unwrap(), None);
        cfg.width = Some(1280);
        cfg.height = Some(720);
        assert_eq!(cfg.target_half().unwrap(), Some((640, 360)));
        cfg.width = Some(1281);
        assert!(cfg.target_half().is_err());
        cfg.width = Some(1280);
        cfg.height = None;
        assert!(cfg.target_half().is_err());
    }

    #[test]
    fn frame_paths_are_sorted_and_suffixed() {
        let dir = std::env::temp_dir();
        let a = dir.join("b.png");
        let b = dir.join("a.png");
        assert!(output_path(false, &b, &dir).to_string_lossy().ends_with("a_x2.png"));
        assert!(output_path(false, &a, &dir).to_string_lossy().ends_with("b_x2.png"));
    }

}
