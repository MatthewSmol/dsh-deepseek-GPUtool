//! Воркер лиц (YuNet + SFace) для плагина GPU tool.
//!
//! Зачем Rust: Python-воркер тратит на кадр ~6,5 мс, но задание на 16 кадров
//! идёт 3,4 с — почти всё время уходит на холодный старт (сессии ONNX Runtime,
//! прогрев cuDNN). Поэтому воркер работает **постоянным процессом**: сессии
//! создаются один раз, а кадры готовятся параллельно и подаются батчами.
//!
//! Режимы:
//!   --serve            постоянный процесс, команды JSONL в stdin
//!   --config <файл>    одноразовый прогон (паритет с python-воркером, тесты)
//!
//! Протокол событий тот же, что у python-воркеров: строки JSON в stdout
//! (`ready|log|progress|result|error`), поэтому движок разбирает их одинаково.

use anyhow::{anyhow, bail, Context, Result};
use ort::session::{Session, SessionOutputs};
use ort::value::Tensor;
use serde::Deserialize;
use serde_json::json;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::time::{Duration, Instant};

/// Эталонные точки 112×112 для выравнивания (как в python/face_worker.py).
const REFERENCE_POINTS: [[f32; 2]; 5] = [
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
];

const NMS_THRESHOLD: f32 = 0.3;

// ---------------------------------------------------------------- конфигурация

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Config {
    /// Каталог с yunet.onnx и sface.onnx.
    model_dir: String,
    /// Видео или изображение-источник (для видео нужен ffmpeg).
    #[serde(default)]
    source: Option<String>,
    /// Эталонное лицо для сравнения.
    #[serde(default)]
    ref_image: Option<String>,
    #[serde(default = "default_fps")]
    fps: f32,
    #[serde(default)]
    start_sec: f32,
    #[serde(default)]
    duration_sec: Option<f32>,
    #[serde(default)]
    width: Option<u32>,
    #[serde(default)]
    height: Option<u32>,
    #[serde(default = "default_conf")]
    conf: f32,
    #[serde(default = "default_match")]
    match_threshold: f32,
    output_dir: String,
    #[serde(default = "default_ffmpeg")]
    ffmpeg_path: String,
    /// Размер входа YuNet; по умолчанию берётся из модели.
    #[serde(default)]
    input_size: Option<u32>,
    /// Сколько кадров подавать одним инференсом (1 — без батчинга).
    #[serde(default = "default_batch")]
    batch: usize,
    /// Потоков на подготовку кадров (resize + blob).
    #[serde(default = "default_threads")]
    threads: usize,
}

fn default_fps() -> f32 {
    1.0
}
fn default_conf() -> f32 {
    0.6
}
fn default_match() -> f32 {
    0.363
}
fn default_batch() -> usize {
    8
}
fn default_threads() -> usize {
    2
}
fn default_ffmpeg() -> String {
    "ffmpeg".to_string()
}

/// Команда постоянного режима.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum CommandMsg {
    Run { id: String, config: Config },
    Ping,
    Shutdown,
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mut mode = String::new();
    let mut config_path: Option<PathBuf> = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--serve" => mode = "serve".into(),
            "--config" => {
                mode = "once".into();
                config_path = args.next().map(PathBuf::from);
            }
            other => bail!("unknown argument: {other}"),
        }
    }
    match mode.as_str() {
        "serve" => serve(),
        "once" => {
            let path = config_path.ok_or_else(|| anyhow!("--config requires a file path"))?;
            let text = std::fs::read_to_string(&path).with_context(|| format!("read {path:?}"))?;
            let config: Config = serde_json::from_str(&text).context("parse config")?;
            run_once(config)
        }
        _ => {
            eprintln!("usage: gputool-face --serve | --config <file>");
            std::process::exit(2);
        }
    }
}

fn emit(value: serde_json::Value) {
    let mut out = std::io::stdout().lock();
    // Ошибка записи в stdout означает, что родитель ушёл: молча завершаемся.
    let _ = serde_json::to_writer(&mut out, &value);
    let _ = out.write_all(b"\n");
    let _ = out.flush();
}

/// Постоянный режим: одна сессия ORT на процесс, задания выполняются по очереди.
fn serve() -> Result<()> {
    let mut engine: Option<Engine> = None;
    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let command: CommandMsg = match serde_json::from_str(&line) {
            Ok(value) => value,
            Err(error) => {
                emit(json!({"type": "error", "code": "bad-command", "message": error.to_string()}));
                continue;
            }
        };
        match command {
            CommandMsg::Ping => emit(json!({"type": "pong"})),
            CommandMsg::Shutdown => break,
            CommandMsg::Run { id, config } => {
                let result = ensure_engine(&mut engine, &config).and_then(|engine| run_job(engine, &id, &config));
                if let Err(error) = result {
                    emit(json!({
                        "type": "error",
                        "id": id,
                        "code": "face-failed",
                        "message": format!("{error:#}"),
                    }));
                }
            }
        }
    }
    Ok(())
}

fn ensure_engine<'a>(slot: &'a mut Option<Engine>, config: &Config) -> Result<&'a mut Engine> {
    if slot.is_none() {
        let engine = Engine::load(config)?;
        emit(json!({
            "type": "ready",
            "engines": engine.describe(),
        }));
        *slot = Some(engine);
    }
    Ok(slot.as_mut().expect("engine"))
}

fn run_once(config: Config) -> Result<()> {
    let mut engine = Engine::load(&config)?;
    emit(json!({"type": "ready", "engines": engine.describe()}));
    run_job(&mut engine, "once", &config)
}

// ------------------------------------------------------------------- движок

struct Engine {
    yunet: Session,
    sface: Session,
    input_w: u32,
    input_h: u32,
    /// Имена выходов по суффиксу шага: (имя, индекс выхода).
    outputs: Vec<(String, usize)>,
    batch: usize,
    threads: usize,
    /// Предвычисленные веса ресайза под размер входа модели.
    resizer: Resizer,
    /// Реально подключённое устройство: "cuda:0" или "cpu".
    device: String,
}

impl Engine {
    fn load(config: &Config) -> Result<Self> {
        let env = ort::init().build().context("init onnxruntime")?;
        let yunet_path = Path::new(&config.model_dir).join("yunet.onnx");
        let sface_path = Path::new(&config.model_dir).join("sface.onnx");
        for path in [&yunet_path, &sface_path] {
            if !path.exists() {
                bail!("model not found: {}", path.display());
            }
        }
        // Устройства перечисляет сам runtime: если CUDA EP реально доступен,
        // сессия получает GPU, иначе остаётся CPU (и это видно в отчёте).
        let cuda_compiled = <ort::ep::CUDA as ort::ep::ExecutionProvider>::is_available().unwrap_or(false);
        let use_cuda = cuda_compiled;
        let build = |path: &PathBuf| -> Result<Session> {
            let builder = Session::builder(&env)?;
            let mut builder = if use_cuda {
                builder
                    .with_execution_providers([ort::ep::CUDA::default().with_device_id(0).build()])
                    .map_err(|error| anyhow!("attach CUDA EP: {error}"))?
            } else {
                builder
            };
            Ok(builder.commit_from_file(path)?)
        };
        let mut yunet = build(&yunet_path)?;
        let mut sface = build(&sface_path)?;
        let device = if use_cuda { "cuda:0" } else { "cpu" };
        let input = yunet.inputs().first().ok_or_else(|| anyhow!("yunet has no inputs"))?;
        let input_w = config.input_size.unwrap_or(640);
        let input_h = config.input_size.unwrap_or(640);
        let _ = input;
        let outputs = yunet
            .outputs()
            .iter()
            .enumerate()
            .map(|(index, outlet)| (outlet.name().to_string(), index))
            .collect::<Vec<_>>();
        // Прогрев: первый запуск инициализирует cuDNN и ядра CUDA.
        let warm = vec![0f32; 3 * input_h as usize * input_w as usize];
        let tensor = Tensor::from_array((
            [1usize, 3, input_h as usize, input_w as usize],
            warm.into_boxed_slice(),
        ))?;
        let _ = yunet.run(ort::inputs![tensor])?;
        let warm_face = vec![0f32; 3 * 112 * 112];
        let tensor = Tensor::from_array((
            [1usize, 3, 112, 112],
            warm_face.into_boxed_slice(),
        ))?;
        let _ = sface.run(ort::inputs![tensor])?;
        Ok(Self {
            yunet,
            sface,
            input_w,
            input_h,
            outputs,
            batch: config.batch.clamp(1, 32),
            threads: config.threads.clamp(1, 8),
            resizer: Resizer::new(1280, 720, input_w, input_h),
            device: device.to_string(),
        })
    }

    fn describe(&self) -> serde_json::Value {
        json!({
            "yunet": "loaded",
            "sface": "loaded",
            "input": [self.input_w, self.input_h],
            "outputs": self.outputs.len(),
            "batch": self.batch,
            "threads": self.threads,
            "device": self.device,
        })
    }
}

// ------------------------------------------------------------------ задание

struct Frame {
    index: usize,
    /// BGR-кадр 1280×720 и далее, как отдаёт ffmpeg.
    rgb: Vec<u8>,
    width: u32,
    height: u32,
}

fn run_job(engine: &mut Engine, id: &str, config: &Config) -> Result<()> {
    std::fs::create_dir_all(&config.output_dir)
        .with_context(|| format!("mkdir {}", config.output_dir))?;
    let jsonl_path = Path::new(&config.output_dir).join("face.jsonl");
    let source = config
        .source
        .as_deref()
        .ok_or_else(|| anyhow!("source is required for the face worker"))?;
    let (width, height) = match (config.width, config.height) {
        (Some(w), Some(h)) => (w, h),
        _ => probe_size(&config.ffmpeg_path, source)?,
    };
    let total = config
        .duration_sec
        .map(|seconds| (seconds * config.fps).round().max(1.0) as usize);

    let mut child = spawn_decoder(config, source)?;
    let stdout = child.stdout.take().ok_or_else(|| anyhow!("ffmpeg stdout"))?;
    let frame_bytes = width as usize * height as usize * 3;
    emit(json!({"type": "log", "id": id, "message": format!(
        "decode: {} fps={} {}x{}", config.ffmpeg_path, config.fps, width, height) }));

    // Читатель отдаёт сырые кадры, подготовка распараллелена ниже.
    let (raw_tx, raw_rx): (SyncSender<Vec<u8>>, Receiver<Vec<u8>>) = sync_channel(engine.threads * 2);
    let reader = std::thread::spawn(move || -> Result<()> {
        let mut stream = BufReader::with_capacity(frame_bytes * 2, stdout);
        loop {
            let mut buffer = vec![0u8; frame_bytes];
            match stream.read_exact(&mut buffer) {
                Ok(()) => {
                    if raw_tx.send(buffer).is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
        Ok(())
    });

    let reference = match &config.ref_image {
        Some(path) => Some(reference_embedding(engine, path, config.conf)?),
        None => None,
    };

    let mut handle = std::io::BufWriter::new(std::fs::File::create(&jsonl_path)?);
    let started = Instant::now();
    let mut frames = 0usize;
    let mut faces_total = 0usize;
    let mut matches = 0usize;
    let mut last_emit = Instant::now();
    let mut pending: Vec<Frame> = Vec::with_capacity(engine.batch);

    loop {
        // Набираем батч: кадры читаются, пока есть, и обрабатываются пачкой.
        pending.clear();
        while pending.len() < engine.batch {
            match raw_rx.recv() {
                Ok(raw) => {
                    frames += 1;
                    pending.push(Frame {
                        index: frames,
                        rgb: raw,
                        width,
                        height,
                    });
                }
                Err(_) => break,
            }
        }
        if pending.is_empty() {
            break;
        }
        let faces = engine.detect_batch(&pending, config.conf)?;
        for (position, frame) in pending.iter().enumerate() {
            let mut rows = Vec::with_capacity(faces[position].len());
            for face in &faces[position] {
                let mut row = json!({
                    "score": round4(face.score),
                    "box": [round2(face.x), round2(face.y), round2(face.w), round2(face.h)],
                    "landmarks": face.landmarks.iter().map(|p| [round2(p[0]), round2(p[1])]).collect::<Vec<_>>(),
                });
                if let Some(reference) = &reference {
                    if let Some(aligned) = align_frame(frame, &face.landmarks) {
                        let embedding = engine.embed(&aligned)?;
                        let similarity = cosine(reference, &embedding);
                        let matched = similarity >= config.match_threshold;
                        row["sim"] = json!(round4(similarity));
                        row["match"] = json!(matched);
                        if matched {
                            matches += 1;
                        }
                    }
                }
                rows.push(row);
            }
            faces_total += faces[position].len();
            writeln!(
                handle,
                "{}",
                json!({
                    "frame": frame.index,
                    "time": round3(frame.index as f32 / config.fps),
                    "faces": rows,
                })
            )?;
        }
        // Прогресс не чаще 4 раз в секунду, чтобы не забивать stdout.
        if last_emit.elapsed() >= Duration::from_millis(400) {
            last_emit = Instant::now();
            let elapsed = started.elapsed().as_secs_f32().max(1e-6);
            let rate = frames as f32 / elapsed;
            let percent = total.map(|total| (frames as f32 * 100.0 / total as f32).min(100.0));
            let eta = total.map(|total| {
                ((total.saturating_sub(frames)) as f32 / rate.max(1e-6) * 1000.0) as u64
            });
            emit(json!({
                "type": "progress",
                "id": id,
                "phase": "faces",
                "percent": percent,
                "frame": frames,
                "totalFrames": total,
                "fps": round2(rate),
                "rate": round3(rate),
                "etaMs": eta,
            }));
        }
    }
    handle.flush().ok();
    let status = child.wait().ok();
    let _ = reader.join();
    let stderr = child_stderr(&mut child);
    if let Some(text) = stderr.as_deref().filter(|text| !text.is_empty()) {
        emit(json!({"type": "log", "id": id, "message": format!("ffmpeg: {text}")}));
    }
    emit(json!({
        "type": "result",
        "id": id,
        "result": {
            "frames": frames,
            "faces": faces_total,
            "matches": matches,
            "jsonl": jsonl_path.to_string_lossy(),
            "engine": "rust",
            "providers": if engine.device == "cpu" {
                json!(["CPUExecutionProvider"])
            } else {
                json!(["CUDAExecutionProvider", "CPUExecutionProvider"])
            },
            "device": engine.device,
            "exitCode": status.and_then(|status| status.code()),
        },
    }));
    Ok(())
}

fn spawn_decoder(config: &Config, source: &str) -> Result<Child> {
    let mut command = Command::new(&config.ffmpeg_path);
    command.args(["-hide_banner", "-loglevel", "error"]);
    if config.start_sec > 0.0 {
        command.args(["-ss", &format!("{}", config.start_sec)]);
    }
    command.args(["-i", source]);
    if let Some(duration) = config.duration_sec {
        command.args(["-t", &format!("{duration}")]);
    }
    command.args([
        "-vf",
        &format!("fps={}", config.fps),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-",
    ]);
    command.stdout(Stdio::piped()).stderr(Stdio::piped()).stdin(Stdio::null());
    command.spawn().with_context(|| format!("spawn {}", config.ffmpeg_path))
}

fn child_stderr(child: &mut Child) -> Option<String> {
    let mut stderr = child.stderr.take()?;
    let mut text = String::new();
    stderr.read_to_string(&mut text).ok()?;
    let trimmed = text.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.chars().take(2000).collect())
    }
}

fn probe_size(ffmpeg: &str, source: &str) -> Result<(u32, u32)> {
    // ffprobe рядом с ffmpeg: /path/ffprobe или ffprobe из PATH.
    let probe = Path::new(ffmpeg)
        .with_file_name("ffprobe")
        .to_string_lossy()
        .to_string();
    let output = Command::new(probe)
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0:s=x",
            source,
        ])
        .output()
        .context("run ffprobe")?;
    let text = String::from_utf8_lossy(&output.stdout);
    let line = text.lines().next().unwrap_or_default();
    let mut parts = line.split('x');
    let width: u32 = parts.next().unwrap_or("0").trim().parse().unwrap_or(0);
    let height: u32 = parts.next().unwrap_or("0").trim().parse().unwrap_or(0);
    if width == 0 || height == 0 {
        bail!("cannot detect frame size for {source}");
    }
    Ok((width, height))
}

// --------------------------------------------------------------- инференс

struct Face {
    score: f32,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    landmarks: [[f32; 2]; 5],
}

impl Engine {
    /// Подаёт кадры батчем; при несовместимости модели с батчем откатывается на 1.
    fn detect_batch(&mut self, frames: &[Frame], conf: f32) -> Result<Vec<Vec<Face>>> {
        let mut result = Vec::with_capacity(frames.len());
        if frames.len() == 1 {
            result.push(self.detect_one(&frames[0], conf)?);
            return Ok(result);
        }
        let mut batched: Vec<Vec<u8>> = Vec::with_capacity(frames.len());
        for frame in frames {
            batched.push(resize_blob(&self.resizer, frame));
        }
        let (width, height) = (self.input_w as usize, self.input_h as usize);
        let mut input = vec![0f32; frames.len() * 3 * height * width];
        for (index, blob) in batched.iter().enumerate() {
            for channel in 0..3 {
                for row in 0..height {
                    local_blob_row(blob, &mut input, index, channel, row, width, height, frames.len());
                }
            }
        }
        let tensor = Tensor::from_array((
            [frames.len(), 3, height, width],
            input.into_boxed_slice(),
        ))?;
        let (input_w, input_h) = (self.input_w, self.input_h);
        let batched_ok = match self.yunet.run(ort::inputs![tensor]) {
            Ok(outputs) => {
                for index in 0..frames.len() {
                    result.push(decode_frame(&outputs, input_w, input_h, frames.len(), index, conf)?);
                }
                true
            }
            Err(_) => false,
        };
        if !batched_ok {
            // Модель без батч-размерности: обрабатываем по одному кадру.
            self.batch = 1;
            result.clear();
            for frame in frames {
                result.push(self.detect_one(frame, conf)?);
            }
        }
        Ok(result)
    }

    fn detect_one(&mut self, frame: &Frame, conf: f32) -> Result<Vec<Face>> {
        let blob = resize_blob(&self.resizer, frame);
        let (width, height) = (self.input_w as usize, self.input_h as usize);
        let mut input = vec![0f32; 3 * height * width];
        for channel in 0..3 {
            for row in 0..height {
                local_blob_row(&blob, &mut input, 0, channel, row, width, height, 1);
            }
        }
        let tensor = Tensor::from_array((
            [1usize, 3, height, width],
            input.into_boxed_slice(),
        ))?;
        let outputs = self.yunet.run(ort::inputs![tensor])?;
        decode_frame(&outputs, self.input_w, self.input_h, 1, 0, conf)
    }

}

/// Декодирует выходы YuNet одного кадра батча в лица после NMS.
fn decode_frame(
    outputs: &SessionOutputs<'_>,
    input_w: u32,
    input_h: u32,
    batch_size: usize,
    batch_index: usize,
    conf: f32,
) -> Result<Vec<Face>> {
    let mut candidates: Vec<Face> = Vec::new();
    for stride in [8usize, 16, 32] {
        let cls = extract_f32(outputs, &format!("cls_{stride}"))?;
        let obj = extract_f32(outputs, &format!("obj_{stride}"))?;
        let bbox = extract_f32(outputs, &format!("bbox_{stride}"))?;
        let kps = extract_f32(outputs, &format!("kps_{stride}"))?;
        let cols = (input_w as usize) / stride;
        let rows = (input_h as usize) / stride;
        let count = (rows * cols).min(cls.len() / batch_size.max(1));
        let base = batch_index * count;
        for index in 0..count {
            let offset = base + index;
            if offset >= cls.len() || offset + 3 >= bbox.len() {
                break;
            }
            let score = (cls[offset] * obj[offset]).clamp(0.0, 1.0).sqrt();
            if score < conf {
                continue;
            }
            let grid_x = (index % cols) as f32;
            let grid_y = (index / cols) as f32;
            let cx = (grid_x + bbox[offset * 4]) * stride as f32;
            let cy = (grid_y + bbox[offset * 4 + 1]) * stride as f32;
            let w = bbox[offset * 4 + 2].exp() * stride as f32;
            let h = bbox[offset * 4 + 3].exp() * stride as f32;
            let mut landmarks = [[0f32; 2]; 5];
            for point in 0..5 {
                let kx = kps[offset * 10 + point * 2];
                let ky = kps[offset * 10 + point * 2 + 1];
                landmarks[point] = [(grid_x + kx) * stride as f32, (grid_y + ky) * stride as f32];
            }
            candidates.push(Face {
                score,
                x: cx - w / 2.0,
                y: cy - h / 2.0,
                w,
                h,
                landmarks,
            });
        }
    }
    Ok(nms(candidates, NMS_THRESHOLD))
}

/// Достаёт выход по имени как плоский массив f32.
fn extract_f32(outputs: &SessionOutputs<'_>, name: &str) -> Result<Vec<f32>> {
    let value = outputs
        .get(name)
        .ok_or_else(|| anyhow!("missing output {name}"))?;
    let (_, data) = value.try_extract_tensor::<f32>()?;
    Ok(data.to_vec())
}

impl Engine {
    fn embed(&mut self, aligned: &[u8]) -> Result<Vec<f32>> {
        let mut input = vec![0f32; 3 * 112 * 112];
        for channel in 0..3 {
            for row in 0..112 {
                for col in 0..112 {
                    let pixel = aligned[(row * 112 + col) * 3 + channel] as f32;
                    input[(channel * 112 + row) * 112 + col] = (pixel - 127.5) / 128.0;
                }
            }
        }
        let tensor = Tensor::from_array(([1usize, 3, 112, 112], input.into_boxed_slice()))?;
        let outputs = self.sface.run(ort::inputs![tensor])?;
        let value = &outputs[0];
        let (_, data) = value.try_extract_tensor::<f32>()?;
        let norm = data.iter().map(|v| v * v).sum::<f32>().sqrt();
        Ok(if norm > 0.0 {
            data.iter().map(|v| v / norm).collect()
        } else {
            data.to_vec()
        })
    }
}

/// Предвычисленные веса билинейного ресайза: без них на кадр уходило больше
/// времени, чем на сам инференс.
struct Resizer {
    width: usize,
    height: usize,
    x0: Vec<usize>,
    x1: Vec<usize>,
    wx: Vec<u32>,
    y0: Vec<usize>,
    y1: Vec<usize>,
    wy: Vec<u32>,
}

impl Resizer {
    fn new(src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> Self {
        let scale_x = src_w as f32 / dst_w as f32;
        let scale_y = src_h as f32 / dst_h as f32;
        let mut x0 = Vec::with_capacity(dst_w as usize);
        let mut x1 = Vec::with_capacity(dst_w as usize);
        let mut wx = Vec::with_capacity(dst_w as usize);
        for col in 0..dst_w as usize {
            let source = ((col as f32 + 0.5) * scale_x - 0.5).max(0.0);
            let left = source.floor() as usize;
            x0.push(left);
            x1.push((left + 1).min(src_w as usize - 1));
            wx.push((((source - left as f32) * 256.0) as u32).min(256));
        }
        let mut y0 = Vec::with_capacity(dst_h as usize);
        let mut y1 = Vec::with_capacity(dst_h as usize);
        let mut wy = Vec::with_capacity(dst_h as usize);
        for row in 0..dst_h as usize {
            let source = ((row as f32 + 0.5) * scale_y - 0.5).max(0.0);
            let top = source.floor() as usize;
            y0.push(top);
            y1.push((top + 1).min(src_h as usize - 1));
            wy.push((((source - top as f32) * 256.0) as u32).min(256));
        }
        Self {
            width: dst_w as usize,
            height: dst_h as usize,
            x0,
            x1,
            wx,
            y0,
            y1,
            wy,
        }
    }
}

/// Копирует одну строку BGR-блоба в NCHW-буфер батча.
#[allow(clippy::too_many_arguments)]
fn local_blob_row(
    blob: &[u8],
    input: &mut [f32],
    batch: usize,
    channel: usize,
    row: usize,
    width: usize,
    height: usize,
    batch_size: usize,
) {
    let src = row * width * 3;
    let dst_base = ((batch * 3 + channel) * height + row) * width;
    for col in 0..width {
        // Вычитание среднего (104,117,123) из cv2.dnn.blobFromImage, порядок BGR.
        let mean = [104.0f32, 117.0, 123.0][channel];
        input[dst_base + col] = blob[src + col * 3 + channel] as f32 - mean;
    }
    let _ = batch_size;
}

/// Билинейный resize по предвычисленным весам (целочисленная арифметика).
fn resize_blob(resizer: &Resizer, frame: &Frame) -> Vec<u8> {
    let (width, height) = (frame.width as usize, frame.height as usize);
    let target_w = resizer.width;
    let target_h = resizer.height;
    let mut out = vec![0u8; target_w * target_h * 3];
    for row in 0..target_h {
        let top = resizer.y0[row] * width;
        let bottom = resizer.y1[row] * width;
        let fy = resizer.wy[row];
        let dst_row = row * target_w;
        for col in 0..target_w {
            let left = resizer.x0[col];
            let right = resizer.x1[col];
            let fx = resizer.wx[col];
            let i00 = (top + left) * 3;
            let i01 = (top + right) * 3;
            let i10 = (bottom + left) * 3;
            let i11 = (bottom + right) * 3;
            let dst = (dst_row + col) * 3;
            for channel in 0..3 {
                let p00 = frame.rgb[i00 + channel] as u32;
                let p01 = frame.rgb[i01 + channel] as u32;
                let p10 = frame.rgb[i10 + channel] as u32;
                let p11 = frame.rgb[i11 + channel] as u32;
                let top_value = ((p00 as i64) << 8) + (p01 as i64 - p00 as i64) * fx as i64 / 256;
                let bottom_value = ((p10 as i64) << 8) + (p11 as i64 - p10 as i64) * fx as i64 / 256;
                let value = top_value + (bottom_value - top_value) * fy as i64 / 256;
                out[dst + channel] = (value >> 8).clamp(0, 255) as u8;
            }
        }
    }
    out
}

/// Greedy NMS по IoU, как cv2.dnn.NMSBoxes.
fn nms(mut faces: Vec<Face>, threshold: f32) -> Vec<Face> {
    faces.sort_by(|left, right| right.score.total_cmp(&left.score));
    let mut kept: Vec<Face> = Vec::new();
    'outer: for face in faces {
        for previous in &kept {
            if iou(&face, previous) > threshold {
                continue 'outer;
            }
        }
        kept.push(face);
    }
    kept
}

fn iou(left: &Face, right: &Face) -> f32 {
    let x1 = left.x.max(right.x);
    let y1 = left.y.max(right.y);
    let x2 = (left.x + left.w).min(right.x + right.w);
    let y2 = (left.y + left.h).min(right.y + right.h);
    let intersection = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let union = left.w * left.h + right.w * right.h - intersection;
    if union <= 0.0 {
        0.0
    } else {
        intersection / union
    }
}

// ------------------------------------------------------------- выравнивание

/// Подобие по 5 точкам (Umeyama без отбраковки): cv2.estimateAffinePartial2D.
fn similarity(src: &[[f32; 2]; 5], dst: &[[f32; 2]; 5]) -> Option<[[f32; 3]; 2]> {
    let n = src.len() as f32;
    let mut src_mean = [0f32; 2];
    let mut dst_mean = [0f32; 2];
    for index in 0..src.len() {
        src_mean[0] += src[index][0] / n;
        src_mean[1] += src[index][1] / n;
        dst_mean[0] += dst[index][0] / n;
        dst_mean[1] += dst[index][1] / n;
    }
    let mut src_var = 0f32;
    let mut cov = [0f32; 2]; // комплексная сумма dst_c * conj(src_c)
    for index in 0..src.len() {
        let sx = src[index][0] - src_mean[0];
        let sy = src[index][1] - src_mean[1];
        let dx = dst[index][0] - dst_mean[0];
        let dy = dst[index][1] - dst_mean[1];
        src_var += sx * sx + sy * sy;
        cov[0] += dx * sx + dy * sy;
        cov[1] += dy * sx - dx * sy;
    }
    if src_var <= 1e-6 {
        return None;
    }
    let norm = cov[0].hypot(cov[1]).max(1e-6);
    let cos = cov[0] / norm;
    let sin = cov[1] / norm;
    // Масштаб: отношение разброса эталона к разбросу найденных точек.
    let mut dst_var = 0f32;
    for index in 0..dst.len() {
        let dx = dst[index][0] - dst_mean[0];
        let dy = dst[index][1] - dst_mean[1];
        dst_var += dx * dx + dy * dy;
    }
    let scale = (dst_var / src_var).sqrt();
    let a = scale * cos;
    let b = scale * sin;
    let tx = dst_mean[0] - (a * src_mean[0] - b * src_mean[1]);
    let ty = dst_mean[1] - (b * src_mean[0] + a * src_mean[1]);
    Some([[a, -b, tx], [b, a, ty]])
}

/// Выравнивает кадр по точкам лица в 112×112 (BGR), как warpAffine.
fn align_frame(frame: &Frame, landmarks: &[[f32; 2]; 5]) -> Option<Vec<u8>> {
    let matrix = similarity(landmarks, &REFERENCE_POINTS)?;
    let (width, height) = (frame.width as usize, frame.height as usize);
    let mut out = vec![0u8; 112 * 112 * 3];
    for row in 0..112usize {
        for col in 0..112usize {
            let x = matrix[0][0] * col as f32 + matrix[0][1] * row as f32 + matrix[0][2];
            let y = matrix[1][0] * col as f32 + matrix[1][1] * row as f32 + matrix[1][2];
            let x0 = x.floor() as isize;
            let y0 = y.floor() as isize;
            if x0 < 0 || y0 < 0 || x0 + 1 >= width as isize || y0 + 1 >= height as isize {
                continue;
            }
            let fx = x - x0 as f32;
            let fy = y - y0 as f32;
            for channel in 0..3 {
                let sample = |px: isize, py: isize| -> f32 {
                    frame.rgb[((py as usize) * width + px as usize) * 3 + channel] as f32
                };
                let top = sample(x0, y0) + (sample(x0 + 1, y0) - sample(x0, y0)) * fx;
                let bottom = sample(x0, y0 + 1) + (sample(x0 + 1, y0 + 1) - sample(x0, y0 + 1)) * fx;
                let value = top + (bottom - top) * fy;
                out[(row * 112 + col) * 3 + channel] = value.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    Some(out)
}

fn cosine(left: &[f32], right: &[f32]) -> f32 {
    let length = left.len().min(right.len());
    let mut dot = 0f32;
    let mut l = 0f32;
    let mut r = 0f32;
    for index in 0..length {
        dot += left[index] * right[index];
        l += left[index] * left[index];
        r += right[index] * right[index];
    }
    if l <= 0.0 || r <= 0.0 {
        0.0
    } else {
        dot / (l.sqrt() * r.sqrt())
    }
}

fn reference_embedding(engine: &mut Engine, path: &str, conf: f32) -> Result<Vec<f32>> {
    let image = read_image(path)?;
    let faces = engine.detect_one(&image, conf)?;
    let face = faces
        .first()
        .ok_or_else(|| anyhow!("no face found in the reference image: {path}"))?;
    let aligned = align_frame(&image, &face.landmarks)
        .ok_or_else(|| anyhow!("cannot align the reference face"))?;
    engine.embed(&aligned)
}

/// Читает PNG/JPEG в BGR через ffmpeg (без зависимости от image-крейтов).
fn read_image(path: &str) -> Result<Frame> {
    let ffmpeg = std::env::var("GPUTOOL_FFMPEG").unwrap_or_else(|_| "ffmpeg".to_string());
    let probe = Command::new(&ffmpeg)
        .args(["-hide_banner", "-loglevel", "error", "-i", path, "-f", "null", "-"])
        .output();
    let _ = probe;
    let size = probe_size(&ffmpeg, path)?;
    let output = Command::new(&ffmpeg)
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            path,
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-",
        ])
        .output()
        .context("decode reference image with ffmpeg")?;
    let rgb = output.stdout;
    let expected = size.0 as usize * size.1 as usize * 3;
    if rgb.len() < expected {
        bail!("cannot decode image {path}");
    }
    Ok(Frame {
        index: 0,
        rgb,
        width: size.0,
        height: size.1,
    })
}

fn round2(value: f32) -> f64 {
    (value as f64 * 100.0).round() / 100.0
}
fn round3(value: f32) -> f64 {
    (value as f64 * 1000.0).round() / 1000.0
}
fn round4(value: f32) -> f64 {
    (value as f64 * 10000.0).round() / 10000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn face(x: f32, y: f32, w: f32, h: f32, score: f32) -> Face {
        Face {
            score,
            x,
            y,
            w,
            h,
            landmarks: [[0.0; 2]; 5],
        }
    }

    #[test]
    fn nms_keeps_the_best_overlapping_box() {
        let kept = nms(
            vec![
                face(0.0, 0.0, 10.0, 10.0, 0.5),
                face(1.0, 1.0, 10.0, 10.0, 0.9),
                face(50.0, 50.0, 10.0, 10.0, 0.7),
            ],
            NMS_THRESHOLD,
        );
        assert_eq!(kept.len(), 2);
        assert!((kept[0].score - 0.9).abs() < 1e-6);
    }

    #[test]
    fn similarity_maps_points_back_to_the_reference() {
        let matrix = similarity(&REFERENCE_POINTS, &REFERENCE_POINTS).expect("matrix");
        assert!((matrix[0][0] - 1.0).abs() < 1e-3);
        assert!(matrix[0][1].abs() < 1e-3);
        assert!(matrix[0][2].abs() < 1e-2);
        assert!((matrix[1][1] - 1.0).abs() < 1e-3);
    }

    #[test]
    fn cosine_is_one_for_equal_vectors() {
        let value = cosine(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]);
        assert!((value - 1.0).abs() < 1e-6);
    }
}
