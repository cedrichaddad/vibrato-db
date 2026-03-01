use crossbeam_channel::{Receiver, Sender};
use directories::ProjectDirs;
use parking_lot::RwLock;
use rtrb::{Consumer, Producer};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use vibrato_core::format_v2::VdbHeaderV2;
use vibrato_core::hnsw::HNSW;
use vibrato_core::metadata::MetadataReader;
use vibrato_core::store::VectorStore;
use vibrato_neural::inference::InferenceEngine;

use crate::commands::{GuiCommand, SearchResult, WorkerResponse};
use crate::dsp_state::DspState;

pub struct VibratoWorker {
    receiver: Receiver<GuiCommand>,
    sender: Sender<WorkerResponse>,
    audio_consumer: Consumer<f32>,
    dsp_state_producer: Producer<DspState>,
    engine: Option<InferenceEngine>,
    searcher: Option<(Arc<VectorStore>, HNSW)>,
    metadata: HashMap<usize, PathBuf>,
    overlay_vectors: Arc<RwLock<Vec<Vec<f32>>>>,
    overlay_hnsw: HNSW,
    overlay_metadata: HashMap<usize, PathBuf>,
}

impl VibratoWorker {
    pub fn new(
        rx: Receiver<GuiCommand>,
        tx: Sender<WorkerResponse>,
        audio_consumer: Consumer<f32>,
        dsp_state_producer: Producer<DspState>,
    ) -> Self {
        let overlay_vectors = Arc::new(RwLock::new(Vec::<Vec<f32>>::new()));
        let overlay_for_hnsw = overlay_vectors.clone();
        let overlay_hnsw = HNSW::new_with_accessor(16, 100, move |id, sink| {
            let guard = overlay_for_hnsw.read();
            let vec = guard
                .get(id)
                .unwrap_or_else(|| panic!("overlay accessor missing vector id={id}"));
            sink(vec);
        });
        Self {
            receiver: rx,
            sender: tx,
            audio_consumer,
            dsp_state_producer,
            engine: None,
            searcher: None,
            metadata: HashMap::new(),
            overlay_vectors,
            overlay_hnsw,
            overlay_metadata: HashMap::new(),
        }
    }

    pub fn spawn(self) {
        std::thread::Builder::new()
            .name("VibratoWorker".into())
            .spawn(move || {
                let mut worker = self;
                // Safety: Catch panics to prevent crashing the plugin (and the DAW)
                if let Err(cause) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    worker.run();
                })) {
                    let msg = format!("VibratoWorker crashed: {cause:?}");
                    eprintln!("{msg}");
                    let _ = worker.sender.send(WorkerResponse::FatalError(msg));
                }
            })
            .expect("Failed to spawn VibratoWorker thread");
    }

    fn run(&mut self) {
        // Initialize Tokio Runtime for async tasks (InferenceEngine)
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("Failed to create Tokio runtime");

        rt.block_on(async {
            // 1. Initialize (Heavy Lifting)
            self.report_status("Initializing...");

            // Resolve paths
            let dirs = ProjectDirs::from("com", "vibrato", "vibrato-vst");
            let data_dir = if let Some(dirs) = &dirs {
                dirs.data_dir().to_path_buf()
            } else {
                PathBuf::from("vibrato_data") // Fallback
            };

            self.report_status(&format!("Data Dir: {:?}", data_dir));

            // Load VectorStore
            self.report_status("Loading Vector Store...");
            let vdb_path = data_dir.join("index.vdb");
            let store = match VectorStore::open(&vdb_path) {
                Ok(s) => {
                    self.report_progress(0.2);
                    Some(Arc::new(s))
                }
                Err(e) => {
                    self.report_status(&format!("Store Error: {}", e));
                    None
                }
            };

            // Load HNSW if store exists
            if let Some(store) = &store {
                self.report_status("Loading HNSW Graph...");
                match File::open(&vdb_path) {
                    Ok(mut file) => {
                        let mut header_bytes = [0u8; 64];
                        let mut loaded = false;

                        if file.read_exact(&mut header_bytes).is_ok() {
                            if let Ok(header) = VdbHeaderV2::from_bytes(&header_bytes) {
                                if header.has_graph() && header.graph_offset > 0 {
                                    if file
                                        .seek(std::io::SeekFrom::Start(header.graph_offset))
                                        .is_ok()
                                    {
                                        let mut reader = BufReader::new(file);
                                        let store_clone = store.clone();
                                        match HNSW::load_from_reader_with_accessor(
                                            &mut reader,
                                            move |id, sink| sink(store_clone.get(id)),
                                        ) {
                                            Ok(hnsw) => {
                                                self.searcher = Some((store.clone(), hnsw));
                                                loaded = true;
                                            }
                                            Err(e) => {
                                                self.report_status(&format!(
                                                    "Graph Load Error: {}",
                                                    e
                                                ));
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        if !loaded {
                            self.report_status("Graph section missing; rebuilding in memory...");
                            let store_clone = store.clone();
                            let mut hnsw = HNSW::new_with_accessor(16, 100, move |id, sink| {
                                sink(store_clone.get(id))
                            });
                            for i in 0..store.count {
                                hnsw.insert(i);
                            }
                            self.searcher = Some((store.clone(), hnsw));
                        }
                    }
                    Err(e) => {
                        self.report_status(&format!("Graph File Error: {}", e));
                    }
                }

                self.report_progress(0.4);
            }

            // Load Inference Engine
            self.report_status("Loading Neural Engine...");
            let model_dir = data_dir.join("models");
            match InferenceEngine::new(&model_dir) {
                Ok(engine) => {
                    self.engine = Some(engine);
                    self.report_progress(0.8);
                }
                Err(e) => {
                    // Non-fatal, we just can't do inference
                    self.report_status(&format!("AI Error: {}", e));
                }
            }

            // Load Metadata from V2 metadata section (single .vdb source of truth).
            self.report_status("Loading Metadata...");
            if let Some(store) = &store {
                if let Some(bytes) = store.metadata_bytes() {
                    match MetadataReader::new(bytes) {
                        Ok(reader) => {
                            if reader.count() == store.count {
                                for i in 0..store.count {
                                    if let Ok(item) = reader.get(i) {
                                        if !item.source_file.is_empty() {
                                            self.metadata
                                                .insert(i, PathBuf::from(item.source_file));
                                        }
                                    }
                                }
                            } else {
                                self.report_status("Metadata count mismatch; using unknown paths");
                            }
                        }
                        Err(e) => {
                            self.report_status(&format!("Metadata parse error: {}", e));
                        }
                    }
                } else {
                    self.report_status("Metadata section missing; using unknown paths");
                }
            }
            self.report_progress(1.0);

            self.report_status("Ready");

            // 2. Event Loop
            // Check for commands or audio
            loop {
                // Non-blocking check for audio first? Or timeout?
                // Use timeout to allow polling audio.
                match self.receiver.recv_timeout(Duration::from_millis(10)) {
                    Ok(cmd) => match cmd {
                        GuiCommand::SearchText(query) => self.handle_text_search(query).await,
                        GuiCommand::Ingest(path) => self.handle_ingest(path).await,
                        GuiCommand::SearchAudio(path) => self.handle_audio_search(path).await,
                        GuiCommand::AnalyzeAudio(samples) => {
                            self.handle_audio_analysis(samples).await
                        }
                    },
                    Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                        // Poll Audio
                        // pop_chunk fails to compile, using simple loop
                        let mut samples_read = 0;
                        while let Ok(_sample) = self.audio_consumer.pop() {
                            // Just drain for now
                            samples_read += 1;
                            if samples_read >= 1024 {
                                break;
                            }
                        }
                        if samples_read > 0 {
                            // We have audio!
                        }
                    }
                    Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
                }
            }
        });
    }

    fn report_status(&self, msg: &str) {
        self.sender
            .send(WorkerResponse::Status(msg.to_string()))
            .ok();
    }

    fn report_progress(&self, progress: f32) {
        self.sender.send(WorkerResponse::Progress(progress)).ok();
    }

    fn overlay_base_id(&self) -> usize {
        self.searcher
            .as_ref()
            .map(|(store, _)| store.count)
            .unwrap_or(0)
    }

    fn insert_overlay_embedding(&mut self, path: PathBuf, embedding: Vec<f32>) {
        let local_id = {
            let mut vectors = self.overlay_vectors.write();
            vectors.push(embedding);
            vectors.len() - 1
        };
        self.overlay_hnsw.insert(local_id);
        self.overlay_metadata.insert(local_id, path);
    }

    fn map_base_results(&self, hits: Vec<(usize, f32)>) -> Vec<SearchResult> {
        hits.into_iter()
            .map(|(id, score)| {
                let path = self
                    .metadata
                    .get(&id)
                    .cloned()
                    .unwrap_or_else(|| PathBuf::from(format!("/unknown/id_{id}")));
                SearchResult { id, path, score }
            })
            .collect()
    }

    fn map_overlay_results(&self, hits: Vec<(usize, f32)>) -> Vec<SearchResult> {
        let base_id = self.overlay_base_id();
        hits.into_iter()
            .map(|(local_id, score)| {
                let path = self
                    .overlay_metadata
                    .get(&local_id)
                    .cloned()
                    .unwrap_or_else(|| PathBuf::from(format!("/overlay/id_{local_id}")));
                SearchResult {
                    id: base_id + local_id,
                    path,
                    score,
                }
            })
            .collect()
    }

    fn search_vector(&mut self, query: &[f32], k: usize, ef: usize) -> Vec<SearchResult> {
        let mut merged = Vec::new();
        if let Some((_store, hnsw)) = &self.searcher {
            merged.extend(self.map_base_results(hnsw.search(query, k, ef)));
        }
        if !self.overlay_vectors.read().is_empty() {
            merged.extend(self.map_overlay_results(self.overlay_hnsw.search(query, k, ef)));
        }
        merged.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        merged.truncate(k);
        merged
    }

    fn complete_search(&mut self, hits: Vec<SearchResult>) {
        let best_state = hits.first().map(|best| DspState::from_score(best.score));
        self.sender.send(WorkerResponse::SearchResults(hits)).ok();
        if let Some(state) = best_state {
            let _ = self.dsp_state_producer.push(state);
        }
    }

    fn collect_audio_files(path: &Path) -> Vec<PathBuf> {
        fn is_supported(path: &Path) -> bool {
            path.extension()
                .and_then(|e| e.to_str())
                .map(|ext| {
                    matches!(
                        ext.to_ascii_lowercase().as_str(),
                        "wav" | "flac" | "mp3" | "ogg" | "m4a" | "aac"
                    )
                })
                .unwrap_or(false)
        }

        if path.is_file() {
            return if is_supported(path) {
                vec![path.to_path_buf()]
            } else {
                Vec::new()
            };
        }

        let mut out = Vec::new();
        let mut stack = vec![path.to_path_buf()];
        while let Some(dir) = stack.pop() {
            let Ok(read_dir) = std::fs::read_dir(&dir) else {
                continue;
            };
            for entry in read_dir.flatten() {
                let p = entry.path();
                if p.is_dir() {
                    stack.push(p);
                } else if is_supported(&p) {
                    out.push(p);
                }
            }
        }
        out
    }

    async fn handle_text_search(&mut self, text: String) {
        if self.engine.is_none() || self.searcher.is_none() {
            self.sender
                .send(WorkerResponse::Error("Engine or Index not loaded".into()))
                .ok();
            return;
        }
        let Some(engine) = self.engine.as_ref().cloned() else {
            return;
        };

        self.report_status("Embedding text...");
        match engine.embed_text(&text).await {
            Ok(vec) => {
                self.report_status("Searching...");
                let hits = self.search_vector(&vec, 10, 64);
                self.complete_search(hits);
                self.report_status("Ready");
            }
            Err(e) => {
                self.sender.send(WorkerResponse::Error(e.to_string())).ok();
                self.report_status("Error");
            }
        }
    }

    async fn handle_audio_search(&mut self, path: PathBuf) {
        if self.engine.is_none() || self.searcher.is_none() {
            self.sender
                .send(WorkerResponse::Error("Engine or Index not loaded".into()))
                .ok();
            return;
        }
        let Some(engine) = self.engine.as_ref().cloned() else {
            return;
        };

        self.report_status("Embedding audio file...");
        match engine.embed_audio_file(&path).await {
            Ok(vec) => {
                self.report_status("Searching...");
                let hits = self.search_vector(&vec, 10, 64);
                self.complete_search(hits);
                self.report_status("Ready");
            }
            Err(e) => {
                self.sender.send(WorkerResponse::Error(e.to_string())).ok();
                self.report_status("Error");
            }
        }
    }

    async fn handle_audio_analysis(&mut self, samples: Vec<f32>) {
        if self.engine.is_none() || self.searcher.is_none() {
            self.sender
                .send(WorkerResponse::Error("Engine or Index not loaded".into()))
                .ok();
            return;
        }
        if samples.is_empty() {
            self.sender
                .send(WorkerResponse::Error("No audio samples provided".into()))
                .ok();
            return;
        }
        let Some(engine) = self.engine.as_ref().cloned() else {
            return;
        };

        match engine.embed_audio(samples).await {
            Ok(vec) => {
                let hits = self.search_vector(&vec, 10, 64);
                self.complete_search(hits);
                self.report_status("Ready");
            }
            Err(e) => {
                self.sender.send(WorkerResponse::Error(e.to_string())).ok();
                self.report_status("Error");
            }
        }
    }

    async fn handle_ingest(&mut self, path: PathBuf) {
        if self.engine.is_none() {
            self.sender
                .send(WorkerResponse::Error("Engine not loaded".into()))
                .ok();
            return;
        }
        let Some(engine) = self.engine.as_ref().cloned() else {
            return;
        };

        let files = Self::collect_audio_files(&path);
        if files.is_empty() {
            self.sender
                .send(WorkerResponse::Error(format!(
                    "No supported audio files found under {}",
                    path.display()
                )))
                .ok();
            return;
        }

        self.report_status("Ingesting audio...");
        let total = files.len().max(1);
        let mut inserted = 0usize;
        for (i, file) in files.iter().enumerate() {
            match engine.embed_audio_file(file).await {
                Ok(vec) => {
                    self.insert_overlay_embedding(file.clone(), vec);
                    inserted += 1;
                }
                Err(e) => {
                    self.sender
                        .send(WorkerResponse::Error(format!(
                            "failed to ingest {}: {}",
                            file.display(),
                            e
                        )))
                        .ok();
                }
            }
            let progress = (i + 1) as f32 / total as f32;
            self.report_progress(progress);
        }

        self.sender
            .send(WorkerResponse::IndexUpdated { count: inserted })
            .ok();
        self.report_status("Ready");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rtrb::RingBuffer;
    use std::time::Duration;

    #[test]
    fn test_worker_boots_and_reports_status() {
        let (gui_tx, worker_rx) = crossbeam_channel::bounded(16);
        let (worker_tx, gui_rx) = crossbeam_channel::bounded(32);
        let (producer, consumer) = RingBuffer::new(1024);
        let (dsp_prod, _dsp_cons) = RingBuffer::new(64);
        drop(producer); // No audio in this test.

        let worker = VibratoWorker::new(worker_rx, worker_tx, consumer, dsp_prod);
        worker.spawn();

        // Worker should emit at least one status/progress message during boot.
        match gui_rx.recv_timeout(Duration::from_secs(1)) {
            Ok(WorkerResponse::Status(_)) | Ok(WorkerResponse::Progress(_)) => {}
            msg => panic!("Expected Starting Status, got {:?}", msg),
        }

        // Disconnect to let the worker terminate its event loop.
        drop(gui_tx);
    }
}
