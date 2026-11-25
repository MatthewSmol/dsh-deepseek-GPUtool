//! Трекинг мяча настольного тенниса: ONNX (TrackNetV2/V3, WASB) через ORT.
//!
//! Кадры идут из ffmpeg (NVDEC при наличии), кольцевой буфер из `frames` кадров
//! складывается в тензор [1, frames*3, H, W]; выход — heatmap, из которой берётся
//! субпиксельный центр. Результат: ball.jsonl/ball.csv и (опционально) видео с
//! точкой, собранное ASS-оверлеем через subtitles-фильтр ffmpeg.

use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use ort::session::Session;
use ort::value::Tensor;
use serde::Deserialize;
use serde_json::{json, Value};

// ------------------------------------------------------------------ конфигурация

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Config {
    source: String,
    model: String,
    output_dir: String,
    #[serde(default)]
    fps: Option<f64>,
    #[serde(default)]
    start_sec: Option<f64>,
    #[serde(default)]
    duration_sec: Option<f64>,
    #[serde(default)]
    threshold: Option<f32>,
    #[serde(default)]
    annotate: Option<bool>,
    #[serde(default)]
    max_gap_frames: Option<usize>,
    #[serde(default)]
    smooth: Option<bool>,
    #[serde(default)]
    max_jump_px: Option<f32>,
    #[serde(default)]
    predict_offset: Option<usize>,
    #[serde(default)]
    output_index: Option<usize>,
    #[serde(default)]
    sigmoid: Option<bool>,
    #[serde(default)]
    normalize: Option<String>,
    #[serde(default)]
    mean: Option<Vec<f32>>,
    #[serde(default)]
    std: Option<Vec<f32>>,
    #[serde(default)]
    frames: Option<usize>,
    #[serde(default)]
    input_width: Option<u32>,
    #[serde(default)]
    input_height: Option<u32>,
    #[serde(default)]
    device: Option<String>,
    #[serde(default)]
    gpu_index: Option<u32>,
    #[serde(default)]
    ffmpeg_path: Option<String>,
    #[serde(default)]
    ffprobe_path: Option<String>,
    #[serde(default)]
    overwrite: Option<bool>,
    /// Одно или несколько изображений вместо видео (1–8 кадров).
    #[serde(default)]
    images: Option<Vec<String>>,
    /// Раскладка входа: "" (кадры), concat (кадры+фон), subtract, subtract_concat.
    #[serde(default)]
    bg_mode: Option<String>,
}

impl Config {
    fn threshold(&self) -> f32 {
        self.threshold.unwrap_or(0.5).clamp(0.0, 1.0)
    }
    fn max_gap(&self) -> usize {
        self.max_gap_frames.unwrap_or(12)
    }
    /// Порог телепорта: по умолчанию 30% ширины кадра за кадр.
    fn max_jump(&self, width: f32) -> f32 {
        self.max_jump_px.unwrap_or(width * 0.3).max(0.0)
    }
    fn smooth(&self) -> bool {
        self.smooth.unwrap_or(true)
    }
    fn offset(&self) -> usize {
        self.predict_offset.unwrap_or(0)
    }
    fn output_index(&self) -> Option<usize> {
        self.output_index
    }
    /// Нормализация входа: WASB обучен на ImageNet mean/std, TrackNet — на 0..1.
    fn normalization(&self) -> Option<([f32; 3], [f32; 3])> {
        if let (Some(mean), Some(std)) = (&self.mean, &self.std) {
            if mean.len() == 3 && std.len() == 3 {
                return Some(([mean[0], mean[1], mean[2]], [std[0], std[1], std[2]]));
            }
        }
        normalize_preset(self.normalize.as_deref(), &self.model)
    }

    /// Модели вроде WASB отдают логиты; auto включает sigmoid, если значения вне [0,1].
    fn needs_sigmoid(&self, heatmap: &[f32]) -> bool {
        match self.sigmoid {
            Some(value) => value,
            None => heatmap.iter().any(|value| *value < 0.0 || *value > 1.0),
        }
    }
    fn frames(&self) -> usize {
        self.frames.unwrap_or(3).clamp(1, 8)
    }
    fn input_width(&self) -> u32 {
        self.input_width.unwrap_or(512)
    }
    fn input_height(&self) -> u32 {
        self.input_height.unwrap_or(288)
    }
    fn annotate(&self) -> bool {
        self.annotate.unwrap_or(true)
    }
    fn bg_mode(&self) -> &str {
        self.bg_mode.as_deref().unwrap_or("")
    }
    /// Число входных каналов под выбранную раскладку и число кадров.
    fn channels(&self, frames: usize) -> usize {
        match self.bg_mode() {
            "concat" => (frames + 1) * 3,
            "subtract" => frames,
            "subtract_concat" => frames * 4,
            _ => frames * 3,
        }
    }
    fn image_paths(&self) -> Vec<String> {
        if let Some(images) = &self.images {
            if !images.is_empty() {
                return images.clone();
            }
        }
        vec![self.source.clone()]
    }
    fn ffmpeg(&self) -> String {
        self.ffmpeg_path.clone().unwrap_or_else(|| "ffmpeg".to_string())
    }
    fn ffprobe(&self) -> String {
        self.ffprobe_path.clone().unwrap_or_else(|| "ffprobe".to_string())
    }
}

// ------------------------------------------------------------------ события

fn emit(value: Value) {
    let stdout = std::io::stdout();
    let mut lock = stdout.lock();
    let _ = writeln!(lock, "{value}");
    let _ = lock.flush();
}

fn emit_log(job: &str, message: impl AsRef<str>) {
    emit(json!({"type": "log", "id": job, "message": message.as_ref()}));
}

fn emit_progress(job: &str, frame: usize, total: usize, started: Instant, phase: &str) {
    let elapsed = started.elapsed().as_secs_f64().max(0.001);
    let fps = frame as f64 / elapsed;
    let percent = (frame as f64 / total.max(1) as f64 * 100.0).min(99.9);
    let eta_ms = if fps > 0.0 && frame > 0 {
        Some(((total.saturating_sub(frame)) as f64 / fps * 1000.0).round() as u64)
    } else {
        None
    };
    emit(json!({
        "type": "progress",
        "id": job,
        "frame": frame,
        "totalFrames": total,
        "fps": (fps * 100.0).round() / 100.0,
        "percent": (percent * 10.0).round() / 10.0,
        "etaMs": eta_ms,
        "phase": phase,
    }));
}

/// Дренаж stderr ffmpeg в отдельном потоке, чтобы заполненный пайп (64 КБ) не
/// блокировал декодер: иначе `wait()` после конца чтения висит навсегда.
fn drain_stderr(mut pipe: impl Read + Send + 'static) -> std::thread::JoinHandle<String> {
    std::thread::spawn(move || {
        let mut buffer = Vec::new();
        let _ = pipe.read_to_end(&mut buffer);
        String::from_utf8_lossy(&buffer).into_owned()
    })
}

// ------------------------------------------------------------------ модель

struct Model {
    session: Session,
    frames: usize,
    channels: usize,
    bg_mode: String,
    width: u32,
    height: u32,
    device: String,
    providers: Vec<String>,
    mean: Option<[f32; 3]>,
    std: Option<[f32; 3]>,
}

impl Model {
    fn load(config: &Config) -> Result<Self> {
        let path = Path::new(&config.model);
        if !path.is_file() {
            bail!("model not found: {}", path.display());
        }
        let width = config.input_width();
        let height = config.input_height();
        let env = ort::init().build().context("init onnxruntime")?;
        let wanted = config.device.as_deref().unwrap_or("auto");
        let mut frames = config.frames();
        let mut last_error: Option<String> = None;
        let mut model: Option<(Session, String, Vec<String>, usize, String)> = None;

        let attempts: Vec<bool> = match wanted {
            "cpu" => vec![false],
            "cuda" => vec![true],
            _ => vec![true, false],
        };
        for use_cuda in attempts {
            let session = if use_cuda {
                match Session::builder(&env).and_then(|builder| {
                    let mut builder = builder
                        .with_execution_providers([ort::ep::CUDA::default().with_device_id(0).build()])
                        .map_err(|error| ort::Error::new(error.to_string()))?;
                    builder.commit_from_file(path)
                }) {
                    Ok(session) => session,
                    Err(error) => {
                        last_error = Some(error.to_string());
                        continue;
                    }
                }
            } else {
                match Session::builder(&env).and_then(|mut builder| builder.commit_from_file(path)) {
                    Ok(session) => session,
                    Err(error) => {
                        last_error = Some(error.to_string());
                        continue;
                    }
                }
            };
            // Подбираем раскладку: concat/subtract или простые frames*3.
            let mut session = session;
            let mut chosen: Option<(usize, usize, String)> = None;
            let config_channels = config.channels(frames);
            let mut candidates: Vec<(usize, usize, String)> = vec![
                (frames, config_channels, config.bg_mode().to_string()),
                (frames, frames * 3, String::new()),
            ];
            for candidate_frames in [1usize, 3, frames] {
                candidates.push((candidate_frames, candidate_frames * 3, String::new()));
                candidates.push((candidate_frames, (candidate_frames + 1) * 3, "concat".to_string()));
            }
            for (candidate, channels, mode) in candidates {
                let probe = vec![0f32; channels * height as usize * width as usize];
                if infer_session(&mut session, channels, width, height, &probe).is_ok() {
                    chosen = Some((candidate, channels, mode));
                    break;
                }
            }
            match chosen {
                Some((candidate, channels, mode)) => {
                    frames = candidate;
                    let device = if use_cuda { "cuda:0".to_string() } else { "cpu".to_string() };
                    let providers = if use_cuda {
                        vec!["CUDAExecutionProvider".to_string(), "CPUExecutionProvider".to_string()]
                    } else {
                        vec!["CPUExecutionProvider".to_string()]
                    };
                    model = Some((session, device, providers, channels, mode));
                    break;
                }
                None => {
                    last_error = Some(format!("model did not accept {}x{} with 3 or 9 channels", width, height));
                }
            }
        }

        let (session, device, providers, channels, bg_mode) =
            model.ok_or_else(|| anyhow!("model-layout-unsupported: {}", last_error.unwrap_or_default()))?;
        if device == "cpu" && wanted != "cpu" {
            emit_log("once", "CUDA is unavailable; ball inference runs on CPU");
        }
        let normalization = config.normalization();
        if normalization.is_some() {
            emit_log("once", "ball: input normalization = ImageNet mean/std");
        }
        Ok(Self {
            session,
            frames,
            channels,
            bg_mode,
            width,
            height,
            device,
            providers,
            mean: normalization.map(|value| value.0),
            std: normalization.map(|value| value.1),
        })
    }

    fn infer(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        infer_session(&mut self.session, self.channels, self.width, self.height, input)
    }
}

fn infer_session(session: &mut Session, channels: usize, width: u32, height: u32, input: &[f32]) -> Result<Vec<f32>> {
    let tensor = Tensor::from_array((
        [1usize, channels, height as usize, width as usize],
        input.to_vec().into_boxed_slice(),
    ))?;
    let outputs = session.run(ort::inputs![tensor])?;
    let value = &outputs[0];
    let (_, data) = value.try_extract_tensor::<f32>()?;
    Ok(data.to_vec())
}

// ------------------------------------------------------------------ ресайз

struct Resizer {
    src_w: usize,
    width: usize,
    height: usize,
    x0: Vec<usize>,
    x1: Vec<usize>,
    wx: Vec<f32>,
    y0: Vec<usize>,
    y1: Vec<usize>,
    wy: Vec<f32>,
}

impl Resizer {
    fn new(src_w: usize, src_h: usize, width: u32, height: u32) -> Self {
        let width = width as usize;
        let height = height as usize;
        let sx = src_w as f64 / width as f64;
        let sy = src_h as f64 / height as f64;
        let axis = |size: usize, src: usize, scale: f64| {
            let mut lo = Vec::with_capacity(size);
            let mut hi = Vec::with_capacity(size);
            let mut weight = Vec::with_capacity(size);
            for position in 0..size {
                let source = ((position as f64 + 0.5) * scale - 0.5).max(0.0);
                let left = source.floor() as usize;
                lo.push(left.min(src - 1));
                hi.push((left + 1).min(src - 1));
                weight.push((source - left as f64) as f32);
            }
            (lo, hi, weight)
        };
        let (x0, x1, wx) = axis(width, src_w, sx);
        let (y0, y1, wy) = axis(height, src_h, sy);
        Self { src_w, width, height, x0, x1, wx, y0, y1, wy }
    }

    fn resize(&self, rgb: &[u8]) -> Vec<f32> {
        let mut planes = vec![0f32; 3 * self.width * self.height];
        for row in 0..self.height {
            let top = self.y0[row] * self.src_w;
            let bottom = self.y1[row] * self.src_w;
            let fy = self.wy[row];
            for col in 0..self.width {
                let i00 = (top + self.x0[col]) * 3;
                let i01 = (top + self.x1[col]) * 3;
                let i10 = (bottom + self.x0[col]) * 3;
                let i11 = (bottom + self.x1[col]) * 3;
                let fx = self.wx[col];
                for channel in 0..3 {
                    let p00 = rgb[i00 + channel] as f32;
                    let p01 = rgb[i01 + channel] as f32;
                    let p10 = rgb[i10 + channel] as f32;
                    let p11 = rgb[i11 + channel] as f32;
                    let top_value = p00 + (p01 - p00) * fx;
                    let bottom_value = p10 + (p11 - p10) * fx;
                    let value = top_value + (bottom_value - top_value) * fy;
                    planes[channel * self.width * self.height + row * self.width + col] = value / 255.0;
                }
            }
        }
        planes
    }
}

// ------------------------------------------------------------------ точки

#[derive(Debug, Clone, Copy)]
struct Point {
    frame: usize,
    time: f64,
    visible: bool,
    x: f32,
    y: f32,
    conf: f32,
}

/// Субпиксельный центр heatmap: argmax + центроид окна 5×5 над max*0.5.
fn heatmap_point(heatmap: &[f32], width: usize, height: usize, threshold: f32) -> (bool, f32, f32, f32) {
    let mut best = 0f32;
    let mut best_index = 0usize;
    for (index, value) in heatmap.iter().enumerate() {
        if *value > best {
            best = *value;
            best_index = index;
        }
    }
    if best < threshold {
        return (false, 0.0, 0.0, best);
    }
    let cx = (best_index % width) as isize;
    let cy = (best_index / width) as isize;
    let cutoff = best * 0.5;
    let mut sum = 0f64;
    let mut sx = 0f64;
    let mut sy = 0f64;
    for dy in -2isize..=2 {
        let y = cy + dy;
        if y < 0 || y >= height as isize {
            continue;
        }
        for dx in -2isize..=2 {
            let x = cx + dx;
            if x < 0 || x >= width as isize {
                continue;
            }
            let value = heatmap[y as usize * width + x as usize];
            if value < cutoff {
                continue;
            }
            let weight = value as f64;
            sum += weight;
            sx += weight * x as f64;
            sy += weight * y as f64;
        }
    }
    if sum <= 0.0 {
        return (true, cx as f32, cy as f32, best);
    }
    (true, (sx / sum) as f32, (sy / sum) as f32, best)
}

fn median(values: &[f32]) -> f32 {
    let mut copy = values.to_vec();
    copy.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    copy[copy.len() / 2]
}

/// Заполняет пропуски и сглаживает траекторию, не трогая короткие обрывы.
fn postprocess(mut points: Vec<Point>, max_gap: usize, smooth: bool, max_jump: f32) -> Vec<Point> {
    // Отбрасываем телепорты: мяч не может перелететь больше max_jump за кадр.
    if max_jump > 0.0 {
        let mut last: Option<(usize, f32, f32)> = None;
        for index in 0..points.len() {
            if !points[index].visible {
                continue;
            }
            if let Some((last_index, last_x, last_y)) = last {
                let distance = ((points[index].x - last_x).powi(2) + (points[index].y - last_y).powi(2)).sqrt();
                let allowed = max_jump * (index - last_index).max(1) as f32;
                if distance > allowed {
                    points[index].visible = false;
                    points[index].conf = 0.0;
                    continue;
                }
            }
            last = Some((index, points[index].x, points[index].y));
        }
    }
    if smooth {
        let visible: Vec<usize> = points.iter().enumerate().filter(|(_, point)| point.visible).map(|(index, _)| index).collect();
        let original = points.clone();
        for &index in &visible {
            let mut xs = Vec::new();
            let mut ys = Vec::new();
            for &other in &visible {
                if other != index && (other as isize - index as isize).abs() <= 2 {
                    xs.push(original[other].x);
                    ys.push(original[other].y);
                }
            }
            if !xs.is_empty() {
                xs.push(original[index].x);
                ys.push(original[index].y);
                points[index].x = median(&xs);
                points[index].y = median(&ys);
            }
        }
    }
    let visible: Vec<usize> = points.iter().enumerate().filter(|(_, point)| point.visible).map(|(index, _)| index).collect();
    if let Some(&first) = visible.first() {
        if first > 0 && first <= max_gap {
            let (x, y) = (points[first].x, points[first].y);
            for index in 0..first {
                points[index].visible = true;
                points[index].x = x;
                points[index].y = y;
                points[index].conf = 0.0;
            }
        }
    }
    if let Some(&last) = visible.last() {
        let tail = points.len().saturating_sub(last + 1);
        if tail > 0 && tail <= max_gap {
            let (x, y) = (points[last].x, points[last].y);
            for index in last + 1..points.len() {
                points[index].visible = true;
                points[index].x = x;
                points[index].y = y;
                points[index].conf = 0.0;
            }
        }
    }
    for pair in visible.windows(2) {
        let (left, right) = (pair[0], pair[1]);
        let gap = right - left - 1;
        if gap > 0 && gap <= max_gap {
            let (x0, y0) = (points[left].x, points[left].y);
            let (x1, y1) = (points[right].x, points[right].y);
            for step in 1..=gap {
                let ratio = step as f32 / (gap + 1) as f32;
                let index = left + step;
                points[index].visible = true;
                points[index].x = x0 + (x1 - x0) * ratio;
                points[index].y = y0 + (y1 - y0) * ratio;
                points[index].conf = 0.0;
            }
        }
    }
    points
}

// ------------------------------------------------------------------ выход

fn write_outputs(config: &Config, points: &[Point]) -> Result<(PathBuf, PathBuf)> {
    let dir = PathBuf::from(&config.output_dir);
    fs::create_dir_all(&dir)?;
    let jsonl_path = dir.join("ball.jsonl");
    let csv_path = dir.join("ball.csv");
    let mut jsonl = BufWriter::new(File::create(&jsonl_path)?);
    let mut csv = BufWriter::new(File::create(&csv_path)?);
    writeln!(csv, "frame,time,visible,x,y,conf")?;
    for point in points {
        writeln!(
            jsonl,
            "{}",
            json!({
                "frame": point.frame,
                "time": (point.time * 1000.0).round() / 1000.0,
                "visible": point.visible,
                "x": (point.x * 100.0).round() / 100.0,
                "y": (point.y * 100.0).round() / 100.0,
                "conf": (point.conf * 1000.0).round() / 1000.0,
            })
        )?;
        writeln!(
            csv,
            "{},{:.3},{},{:.2},{:.2},{:.3}",
            point.frame, point.time, if point.visible { 1 } else { 0 }, point.x, point.y, point.conf
        )?;
    }
    Ok((jsonl_path, csv_path))
}

fn build_ass(config: &Config, points: &[Point], width: u32, height: u32, fps: f64) -> Result<PathBuf> {
    let dir = PathBuf::from(&config.output_dir);
    let path = dir.join("ball.ass");
    let font_size = (height as f64 / 22.0).clamp(14.0, 90.0);
    let mut file = BufWriter::new(File::create(&path)?);
    writeln!(file, "[Script Info]")?;
    writeln!(file, "ScriptType: v4.00+")?;
    writeln!(file, "PlayResX: {width}")?;
    writeln!(file, "PlayResY: {height}")?;
    writeln!(file, "WrapStyle: 2")?;
    writeln!(file)?;
    writeln!(file, "[V4+ Styles]")?;
    writeln!(file, "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding")?;
    writeln!(
        file,
        "Style: Dot,DejaVu Sans,{font_size:.0},&H0000FFFF,&H0000FFFF,&H00000000,&H64000000,1,0,0,0,100,100,0,0,1,2,1,5,0,0,0,1"
    )?;
    writeln!(file)?;
    writeln!(file, "[Events]")?;
    writeln!(file, "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text")?;
    let duration_cs = ((100.0 / fps.max(1.0)).round() as i64).max(1);
    for point in points {
        if !point.visible {
            continue;
        }
        let start_cs = (point.frame as f64 / fps * 100.0).round() as i64;
        let end_cs = start_cs + duration_cs;
        writeln!(
            file,
            "Dialogue: 0,{},{},Dot,,0,0,0,,{{\\pos({:.1},{:.1})}}●",
            ass_time(start_cs),
            ass_time(end_cs),
            point.x,
            point.y
        )?;
    }
    Ok(path)
}

fn ass_time(centiseconds: i64) -> String {
    let total = centiseconds.max(0);
    let cs = total % 100;
    let seconds = total / 100;
    let s = seconds % 60;
    let minutes = seconds / 60;
    let m = minutes % 60;
    let hours = minutes / 60;
    format!("{hours}:{m:02}:{s:02}.{cs:02}")
}

fn annotate(config: &Config, ass: &Path, source: &Path, output: &Path, fps: f64, width: u32, height: u32, job: &str, started: Instant, total: usize) -> Result<()> {
    let gpu = config.gpu_index.unwrap_or(0);
    let mut encoder_args: Vec<String> = Vec::new();
    encoder_args.extend([
        "-hide_banner".into(), "-nostdin".into(), "-y".into(), "-loglevel".into(), "error".into(),
        "-progress".into(), "pipe:1".into(), "-nostats".into(),
    ]);
    if let Some(start) = config.start_sec.filter(|value| *value > 0.0) {
        encoder_args.extend(["-ss".into(), format!("{start:.3}")]);
    }
    if let Some(duration) = config.duration_sec.filter(|value| *value > 0.0) {
        encoder_args.extend(["-t".into(), format!("{duration:.3}")]);
    }
    encoder_args.extend([
        "-i".into(), source.to_string_lossy().into_owned(),
        "-vf".into(),
        format!("subtitles={}", escape_filter_path(ass)),
        "-c:v".into(), "hevc_nvenc".into(),
        "-gpu".into(), gpu.to_string(),
        "-preset".into(), "p5".into(), "-tune".into(), "hq".into(), "-rc".into(), "vbr".into(), "-cq".into(), "22".into(),
        "-tag:v".into(), "hvc1".into(),
        "-c:a".into(), "copy".into(),
        "-movflags".into(), "+faststart".into(),
        "-s".into(), format!("{width}x{height}"),
        output.to_string_lossy().into_owned(),
    ]);
    let mut command = Command::new(config.ffmpeg());
    command.args(&encoder_args);
    // Дочерний ffmpeg видит все GPU: CUDA_VISIBLE_DEVICES действует только на ORT.
    command.env_remove("CUDA_VISIBLE_DEVICES");
    command.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = command.spawn().context("cannot start annotate encoder")?;
    if let Some(stdout) = child.stdout.take() {
        let _duration = config.duration_sec.unwrap_or(0.0).max(1.0);
        for line in BufReader::new(stdout).lines().flatten() {
            if let Some(value) = line.strip_prefix("out_time_us=") {
                if let Ok(micros) = value.trim().parse::<f64>() {
                    let seconds = micros / 1_000_000.0;
                    let frame = ((seconds * fps).round() as usize).min(total);
                    emit_progress(job, frame, total, started, "annotate");
                }
            }
        }
    }
    let status = child.wait().context("annotate wait")?;
    if !status.success() {
        let mut message = String::new();
        if let Some(mut stderr) = child.stderr.take() {
            let _ = stderr.read_to_string(&mut message);
        }
        bail!("encode-failed: {}", message.trim());
    }
    Ok(())
}

fn escape_filter_path(path: &Path) -> String {
    let text = path.to_string_lossy().replace('\\', "\\\\").replace(':', "\\:").replace('\'', "\\'");
    format!("'{text}'")
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
        bail!("decoder-failed: ffprobe: {}", String::from_utf8_lossy(&output.stderr).trim());
    }
    let probe: Probe = serde_json::from_slice(&output.stdout).context("bad ffprobe json")?;
    let video = probe
        .streams
        .iter()
        .find(|stream| stream.codec_type.as_deref() == Some("video") && stream.width.unwrap_or(0) > 0)
        .ok_or_else(|| anyhow!("decoder-failed: no video stream"))?;
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

fn run_job(job: &str, config: &Config, model: &mut Option<Model>) -> Result<Value> {
    let started = Instant::now();
    let source = Path::new(&config.source);
    if !source.is_file() {
        bail!("input-missing: {}", source.display());
    }
    if model.is_none() {
        let loaded = Model::load(config)?;
        emit_log(
            job,
            format!(
                "ball model: {}x{} frames={} offset={} device={}",
                loaded.width, loaded.height, loaded.frames, config.offset(), loaded.device
            ),
        );
        *model = Some(loaded);
    }
    let engine = model.as_mut().expect("model");
    if config.images.is_some() || is_image_path(source) {
        return run_image(job, config, engine, &started);
    }
    if engine.channels != engine.frames * 3 {
        bail!("model-layout-unsupported: bg_mode={} is supported for images only", engine.bg_mode);
    }
    let (width, height, source_fps, source_duration, has_audio) = probe_video(config, source)?;
    let fps = config.fps.unwrap_or(source_fps).clamp(1.0, 60.0);
    let start = config.start_sec.unwrap_or(0.0).max(0.0);
    let duration = config
        .duration_sec
        .unwrap_or_else(|| (source_duration - start).max(0.0));
    let total_frames = ((duration * fps).round() as usize).max(1);

    let dir = PathBuf::from(&config.output_dir);
    fs::create_dir_all(&dir)?;
    let video_path = dir.join("ball.mp4");
    if video_path.exists() && config.overwrite != Some(true) {
        bail!("output-exists: {}", video_path.display());
    }

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
        "-vf".into(), format!("fps={fps}"),
        "-f".into(), "rawvideo".into(), "-pix_fmt".into(), "rgb24".into(), "-".into(),
    ]);
    let mut decoder = Command::new(config.ffmpeg())
        .args(&decoder_args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("decoder-failed: cannot start ffmpeg")?;
    let mut decoder_out = decoder.stdout.take().context("decoder stdout")?;
    let decoder_err = decoder.stderr.take().map(drain_stderr);

    let resizer = Resizer::new(width as usize, height as usize, engine.width, engine.height);
    let frame_bytes = width as usize * height as usize * 3;
    let plane = engine.width as usize * engine.height as usize;
    let mut raw = vec![0u8; frame_bytes];
    let mut buffer: Vec<Vec<f32>> = Vec::with_capacity(engine.frames);
    let channels = engine.frames * 3;
    let mut tensor = vec![0f32; channels * plane];
    let mut points: Vec<Point> = Vec::with_capacity(total_frames);
    let mut frame_index = 0usize;
    let mut sigmoid_reported = false;
    let mut last_emit = Instant::now();

    loop {
        match decoder_out.read_exact(&mut raw) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(error) => bail!("decoder-failed: {error}"),
        }
        let resized = resizer.resize(&raw);
        if buffer.len() < engine.frames {
            buffer.push(resized);
        } else {
            buffer.remove(0);
            buffer.push(resized);
        }
        if buffer.len() == engine.frames {
            for (slot, frame) in buffer.iter().enumerate() {
                for channel in 0..3 {
                    let from = channel * plane;
                    let to = (slot * 3 + channel) * plane;
                    match (engine.mean, engine.std) {
                        (Some(mean), Some(std)) => {
                            for index in 0..plane {
                                tensor[to + index] = (frame[from + index] - mean[channel]) / std[channel];
                            }
                        }
                        _ => tensor[to..to + plane].copy_from_slice(&frame[from..from + plane]),
                    }
                }
            }
            let output = engine.infer(&tensor)?;
            if output.len() % plane != 0 {
                bail!(
                    "model-layout-unsupported: {} values is not a multiple of {plane} (H*W)",
                    output.len()
                );
            }
            let heatmaps = output.len() / plane;
            let channel = match config.output_index() {
                Some(index) if index < heatmaps => index,
                Some(index) => bail!(
                    "model-layout-unsupported: outputIndex={index}, but the model returns {heatmaps} heatmaps"
                ),
                None => heatmaps.saturating_sub(1),
            };
            let raw_heatmap = &output[channel * plane..(channel + 1) * plane];
            let use_sigmoid = config.needs_sigmoid(raw_heatmap);
            if use_sigmoid && !sigmoid_reported {
                emit_log(job, "ball: heatmap looks like logits, applying sigmoid");
                sigmoid_reported = true;
            }
            let heatmap: Vec<f32> = if use_sigmoid {
                raw_heatmap.iter().map(|value| 1.0 / (1.0 + (-value).exp())).collect()
            } else {
                raw_heatmap.to_vec()
            };
            let (visible, x, y, conf) = heatmap_point(&heatmap, engine.width as usize, engine.height as usize, config.threshold());
            let predicted = frame_index.saturating_sub(config.offset());
            points.push(Point {
                frame: predicted,
                time: predicted as f64 / fps,
                visible,
                x: x * width as f32 / engine.width as f32,
                y: y * height as f32 / engine.height as f32,
                conf,
            });
        }
        frame_index += 1;
        if last_emit.elapsed().as_millis() >= 500 {
            emit_progress(job, frame_index.min(total_frames), total_frames, started, "track");
            last_emit = Instant::now();
        }
    }
    // Не останавливаемся по счётчику: лишние кадры на границе `-t`/fps иначе
    // оставляют декодер заблокированным на пайпе, а wait() висит навсегда.
    drop(decoder_out);
    emit_progress(job, frame_index.min(total_frames), total_frames, started, "finalize");
    let decoder_status = decoder.wait().context("decoder wait")?;
    let decoder_log = decoder_err.map(|handle| handle.join().unwrap_or_default()).unwrap_or_default();
    if !decoder_status.success() {
        if points.is_empty() {
            bail!("decoder-failed: {}", decoder_log.trim());
        }
        emit_log(job, format!("decoder warning: {}", decoder_log.trim()));
    }

    // Добиваем пропущенные в начале кадры, чтобы jsonl был по всему таймлайну.
    let mut by_frame: Vec<Option<Point>> = vec![None; total_frames];
    for point in points {
        if point.frame < total_frames {
            by_frame[point.frame] = Some(point);
        }
    }
    let mut ordered: Vec<Point> = by_frame
        .into_iter()
        .enumerate()
        .map(|(index, point)| point.unwrap_or(Point {
            frame: index,
            time: index as f64 / fps,
            visible: false,
            x: 0.0,
            y: 0.0,
            conf: 0.0,
        }))
        .collect();
    ordered = postprocess(ordered, config.max_gap(), config.smooth(), config.max_jump(width as f32));
    let detected = ordered.iter().filter(|point| point.visible).count();
    let (jsonl_path, csv_path) = write_outputs(config, &ordered)?;
    let ass_path = build_ass(config, &ordered, width, height, fps)?;

    let video = if config.annotate() {
        emit_log(job, format!("annotate: {} visible points", detected));
        annotate(config, &ass_path, source, &video_path, fps, width, height, job, started, total_frames)?;
        Some(video_path.to_string_lossy().into_owned())
    } else {
        None
    };

    Ok(json!({
        "engine": "rust",
        "device": engine.device,
        "providers": engine.providers,
        "frames": total_frames,
        "detected": detected,
        "fps": fps,
        "hasAudio": has_audio,
        "jsonl": jsonl_path.to_string_lossy(),
        "csv": csv_path.to_string_lossy(),
        "ass": ass_path.to_string_lossy(),
        "video": video,
        "model": {
            "frames": engine.frames,
            "width": engine.width,
            "height": engine.height,
            "offset": config.offset(),
        },
        "elapsedMs": started.elapsed().as_millis(),
    }))
}

fn is_image_path(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|value| value.to_str()).map(|value| value.to_ascii_lowercase()).as_deref(),
        Some("png" | "jpg" | "jpeg" | "webp" | "bmp")
    )
}

fn load_image(path: &Path) -> Result<(u32, u32, Vec<u8>)> {
    let image = image::open(path).with_context(|| format!("image-decode-failed: {}", path.display()))?;
    let image = image.to_rgb8();
    let (width, height) = image.dimensions();
    Ok((width, height, image.into_raw()))
}

fn draw_marker(path: &Path, x: f32, y: f32, output: &Path) -> Result<()> {
    let mut image = image::open(path).with_context(|| format!("image-decode-failed: {}", path.display()))?.to_rgb8();
    let (width, height) = image.dimensions();
    let radius = 14i32.max((width.min(height) as i32) / 80);
    let cx = x.round() as i32;
    let cy = y.round() as i32;
    for angle in 0..720 {
        let theta = angle as f32 * std::f32::consts::PI / 360.0;
        for thickness in 0..2 {
            let px = cx + ((radius - thickness) as f32 * theta.cos()).round() as i32;
            let py = cy + ((radius - thickness) as f32 * theta.sin()).round() as i32;
            if px >= 0 && py >= 0 && px < width as i32 && py < height as i32 {
                image.put_pixel(px as u32, py as u32, image::Rgb([255, 32, 32]));
            }
        }
    }
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    image.save(output).with_context(|| format!("encode-failed: {}", output.display()))?;
    Ok(())
}

/// Захват мяча на одном изображении (или 1–8 кадрах): без ffmpeg.
fn run_image(job: &str, config: &Config, engine: &mut Model, started: &Instant) -> Result<Value> {
    let paths: Vec<PathBuf> = config.image_paths().into_iter().map(PathBuf::from).collect();
    if paths.is_empty() {
        bail!("input-missing: images are empty");
    }
    let mut frames_rgb: Vec<(u32, u32, Vec<u8>)> = Vec::with_capacity(paths.len());
    for path in &paths {
        if !path.is_file() {
            bail!("input-missing: {}", path.display());
        }
        frames_rgb.push(load_image(path)?);
    }
    let (src_w, src_h) = (frames_rgb[0].0, frames_rgb[0].1);
    let mut resized: Vec<Vec<f32>> = Vec::with_capacity(frames_rgb.len());
    for (width, height, rgb) in &frames_rgb {
        if *width != src_w || *height != src_h {
            bail!("image-size-mismatch: all images must have the same size");
        }
        resized.push(Resizer::new(*width as usize, *height as usize, engine.width, engine.height).resize(rgb));
    }
    let plane = engine.width as usize * engine.height as usize;
    let frames = engine.frames;
    let mut tensor = vec![0f32; engine.channels * plane];
    for slot in 0..frames {
        let frame = &resized[slot.min(resized.len() - 1)];
        if engine.channels == frames * 3 || engine.channels == (frames + 1) * 3 {
            for channel in 0..3 {
                let from = channel * plane;
                let to = (slot * 3 + channel) * plane;
                match (engine.mean, engine.std) {
                    (Some(mean), Some(std)) => {
                        for index in 0..plane {
                            tensor[to + index] = (frame[from + index] - mean[channel]) / std[channel];
                        }
                    }
                    _ => tensor[to..to + plane].copy_from_slice(&frame[from..from + plane]),
                }
            }
        } else if engine.channels == frames {
            // subtract: разница с первым кадром как фоном.
            for index in 0..plane {
                tensor[slot * plane + index] = frame[index] - resized[0][index];
            }
        } else {
            bail!("model-layout-unsupported: bg_mode={} channels={}", engine.bg_mode, engine.channels);
        }
    }
    if engine.channels == (frames + 1) * 3 {
        let background = &resized[0];
        for channel in 0..3 {
            let from = channel * plane;
            let to = (frames * 3 + channel) * plane;
            tensor[to..to + plane].copy_from_slice(&background[from..from + plane]);
        }
    }
    emit_progress(job, 1, 1, *started, "infer");
    let output = engine.infer(&tensor)?;
    if output.len() % plane != 0 {
        bail!("model-layout-unsupported: {} values is not a multiple of {plane}", output.len());
    }
    let heatmaps = output.len() / plane;
    let channel = match config.output_index() {
        Some(index) if index < heatmaps => index,
        Some(index) => bail!("model-layout-unsupported: outputIndex={index}, but the model returns {heatmaps} heatmaps"),
        None => heatmaps.saturating_sub(1),
    };
    let raw_heatmap = &output[channel * plane..(channel + 1) * plane];
    let use_sigmoid = config.needs_sigmoid(raw_heatmap);
    let heatmap: Vec<f32> = if use_sigmoid {
        raw_heatmap.iter().map(|value| 1.0 / (1.0 + (-value).exp())).collect()
    } else {
        raw_heatmap.to_vec()
    };
    let (visible, hx, hy, conf) = heatmap_point(&heatmap, engine.width as usize, engine.height as usize, config.threshold());
    let x = hx * src_w as f32 / engine.width as f32;
    let y = hy * src_h as f32 / engine.height as f32;
    let dir = PathBuf::from(&config.output_dir);
    fs::create_dir_all(&dir)?;
    let stem = paths[0].file_stem().and_then(|value| value.to_str()).unwrap_or("image");
    let annotated = dir.join(format!("{stem}_ball.png"));
    if config.annotate() && visible {
        draw_marker(&paths[0], x, y, &annotated)?;
    }
    emit_progress(job, 1, 1, *started, "done");
    Ok(json!({
        "engine": "rust",
        "device": engine.device,
        "providers": engine.providers,
        "images": paths.len(),
        "width": src_w,
        "height": src_h,
        "x": x,
        "y": y,
        "confidence": conf,
        "visible": visible,
        "annotated": if config.annotate() && visible { Some(annotated.to_string_lossy().to_string()) } else { None },
        "model": {
            "frames": engine.frames,
            "channels": engine.channels,
            "bgMode": engine.bg_mode,
            "width": engine.width,
            "height": engine.height,
        },
        "elapsedMs": started.elapsed().as_millis(),
    }))
}

fn classify(error: &anyhow::Error) -> &'static str {
    let text = error.to_string();
    if text.starts_with("model not found") {
        "ball-model-missing"
    } else if text.starts_with("input-missing") {
        "input-missing"
    } else if text.starts_with("model-layout-unsupported") {
        "model-layout-unsupported"
    } else if text.starts_with("output-exists") {
        "output-exists"
    } else if text.starts_with("decoder-failed") {
        "decode-failed"
    } else if text.starts_with("encode-failed") {
        "encode-failed"
    } else if text.contains("CUDA") {
        "cuda-ep-unavailable"
    } else {
        "ball-failed"
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
    let config_path = value("--config").ok_or_else(|| anyhow!("--config or --serve is required"))?;
    let text = fs::read_to_string(&config_path)?;
    let config: Config = serde_json::from_str(&text).context("bad config json")?;
    let mut model: Option<Model> = None;
    match run_job("once", &config, &mut model) {
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
    let mut model: Option<Model> = None;
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
                match run_job(&job, &config, &mut model) {
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

/// Пресет нормализации по имени модели: WASB — ImageNet, остальные — 0..1.
fn normalize_preset(override_name: Option<&str>, model: &str) -> Option<([f32; 3], [f32; 3])> {
    let imagenet = ([0.485f32, 0.456, 0.406], [0.229f32, 0.224, 0.225]);
    match override_name {
        Some("none") => None,
        Some("imagenet") => Some(imagenet),
        _ => {
            let name = model.to_ascii_lowercase();
            if name.contains("wasb") || name.contains("blurball") {
                Some(imagenet)
            } else {
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heatmap_point_finds_subpixel_center_and_threshold() {
        let (w, h) = (16usize, 9usize);
        let mut heatmap = vec![0f32; w * h];
        heatmap[4 * w + 7] = 0.9;
        heatmap[4 * w + 6] = 0.45;
        let (visible, x, y, conf) = heatmap_point(&heatmap, w, h, 0.5);
        assert!(visible);
        assert!((x - 6.66).abs() < 0.2, "x={x}");
        assert!((y - 4.0).abs() < 0.01);
        assert!((conf - 0.9).abs() < 0.001);
        let (hidden, _, _, _) = heatmap_point(&heatmap, w, h, 0.95);
        assert!(!hidden);
    }

    #[test]
    fn postprocess_fills_short_gaps_and_keeps_long_ones() {
        let point = |frame: usize, visible: bool| Point { frame, time: frame as f64 / 30.0, visible, x: frame as f32, y: 0.0, conf: 1.0 };
        let points = vec![
            point(0, true), point(1, false), point(2, false), point(3, true),
            point(4, false), point(5, false), point(6, false),
        ];
        let filled = postprocess(points, 2, false, 0.0);
        // короткий разрыв между детекциями заполняется, длинный хвост — нет
        assert!(filled[1].visible && filled[2].visible);
        assert!((filled[1].x - 1.0).abs() < 0.001 && (filled[2].x - 2.0).abs() < 0.001);
        assert!(!filled[4].visible && !filled[5].visible && !filled[6].visible);
    }

    #[test]
    fn multiple_heatmaps_pick_the_requested_channel() {
        // Три heatmap: пики в разных местах; последний канал — самый правый.
        let (w, h) = (8usize, 4usize);
        let plane = w * h;
        let mut output = vec![0f32; 3 * plane];
        output[1 * w + 1] = 0.9; // канал 0 -> x=1
        output[plane + 2 * w + 5] = 0.9; // канал 1 -> x=5
        output[2 * plane + 3 * w + 7] = 0.9; // канал 2 -> x=7
        let pick = |index: usize| {
            let heatmap = &output[index * plane..(index + 1) * plane];
            heatmap_point(heatmap, w, h, 0.5)
        };
        assert_eq!(pick(0).1 as usize, 1);
        assert_eq!(pick(2).1 as usize, 7);
        assert_eq!(3 * plane / plane, 3);
    }

    #[test]
    fn normalization_preset_depends_on_the_model_family() {
        assert!(normalize_preset(None, "/models/wasb_tennis.onnx").is_some());
        assert!(normalize_preset(None, "/models/tracknetv2.onnx").is_none());
        assert!(normalize_preset(Some("imagenet"), "/models/tracknetv2.onnx").is_some());
        assert!(normalize_preset(Some("none"), "/models/wasb_tennis.onnx").is_none());
    }

    #[test]
    fn logits_are_detected_and_sigmoid_maps_them_to_probabilities() {
        let logits = vec![-10.0f32, 0.0, 2.0];
        let needs = logits.iter().any(|value| *value < 0.0 || *value > 1.0);
        assert!(needs);
        let probabilities: Vec<f32> = logits.iter().map(|value| 1.0 / (1.0 + (-value).exp())).collect();
        assert!((probabilities[1] - 0.5).abs() < 1e-6);
        assert!(probabilities[2] > 0.85 && probabilities[0] < 0.001);
    }

    #[test]
    fn jumps_above_the_limit_are_dropped_before_gap_filling() {
        let point = |frame: usize, x: f32| Point { frame, time: frame as f64 / 25.0, visible: true, x, y: 0.0, conf: 1.0 };
        let points = vec![point(0, 100.0), point(1, 110.0), point(2, 900.0), point(3, 130.0)];
        let filtered = postprocess(points, 0, false, 100.0);
        assert!(filtered[0].visible && filtered[1].visible && filtered[3].visible);
        assert!(!filtered[2].visible);
    }

    #[test]
    fn ass_time_and_events_are_valid() {
        assert_eq!(ass_time(0), "0:00:00.00");
        assert_eq!(ass_time(1234), "0:00:12.34");
        let points = vec![Point { frame: 0, time: 0.0, visible: true, x: 10.0, y: 20.0, conf: 1.0 }];
        let config = Config {
            images: None,
            bg_mode: None,
            source: "/tmp/x.mp4".into(), model: "/tmp/x.onnx".into(), output_dir: std::env::temp_dir().join("ball-test").to_string_lossy().into(),
            fps: None, start_sec: None, duration_sec: None, threshold: None, annotate: None, max_gap_frames: None,
            smooth: None, max_jump_px: None, predict_offset: None, output_index: None, sigmoid: None, normalize: None, mean: None, std: None, frames: None, input_width: None, input_height: None, device: None,
            gpu_index: None, ffmpeg_path: None, ffprobe_path: None, overwrite: None,
        };
        let _ = fs::create_dir_all(&config.output_dir);
        let path = build_ass(&config, &points, 320, 180, 30.0).unwrap();
        let text = fs::read_to_string(path).unwrap();
        assert!(text.contains("Dialogue: 0,0:00:00.00,0:00:00.03,Dot,,0,0,0,,{\\pos(10.0,20.0)}●"));
    }
}
