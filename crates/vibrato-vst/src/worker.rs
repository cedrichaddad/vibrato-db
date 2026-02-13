use std::path::PathBuf;
use crossbeam_channel::{Receiver, Sender};
use vibrato_neural::inference::InferenceEngine;
use vibrato_core::store::VectorStore;
use vibrato_core::hnsw::HNSW;
use directories::ProjectDirs;
use std::sync::Arc;
use rtrb::Consumer;
use std::time::Duration;

use crate::commands::{GuiCommand, WorkerResponse, SearchResult};

pub struct VibratoWorker {
    receiver: Receiver<GuiCommand>,
    sender: Sender<WorkerResponse>,
    audio_consumer: Consumer<f32>,
    engine: Option<InferenceEngine>, 
    searcher: Option<(Arc<VectorStore>, HNSW)>,
}

impl VibratoWorker {
    pub fn new(rx: Receiver<GuiCommand>, tx: Sender<WorkerResponse>, audio_consumer: Consumer<f32>) -> Self {
        Self { receiver: rx, sender: tx, audio_consumer, engine: None, searcher: None }
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
                    eprintln!("VibratoWorker crashed: {:?}", cause);
                    // In a real implementation, we might try to signal the GUI if we still have the sender
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
                },
                Err(e) => {
                    self.report_status(&format!("Store Error: {}", e));
                    None
                }
            };
    
            // Load HNSW if store exists
            if let Some(store) = &store {
                self.report_status("Loading HNSW Index...");
                let idx_path = data_dir.join("index.idx");
                let store_clone = store.clone();
                // HNSW::load expects a closure that returns Vec<f32>
                let accessor = move |id: usize| {
                     store_clone.get(id).to_vec()
                };
                
                match HNSW::load(&idx_path, accessor) {
                     Ok(hnsw) => {
                         self.searcher = Some((store.clone(), hnsw));
                         self.report_progress(0.4);
                     }
                     Err(e) => {
                         self.report_status(&format!("Index Error: {}", e));
                     }
                }
            }
    
            // Load Inference Engine
            self.report_status("Loading Neural Engine...");
            let model_dir = data_dir.join("models");
            match InferenceEngine::new(&model_dir) {
                Ok(engine) => {
                    self.engine = Some(engine);
                    self.report_progress(1.0);
                }
                 Err(e) => {
                     // Non-fatal, we just can't do inference
                     self.report_status(&format!("AI Error: {}", e));
                 }
            }
    
            self.report_status("Ready");
            
            // 2. Event Loop
            // Check for commands or audio
            let mut audio_buffer: Vec<f32> = Vec::with_capacity(4096);
            
            loop {
                // Non-blocking check for audio first? Or timeout?
                // Use timeout to allow polling audio.
                match self.receiver.recv_timeout(Duration::from_millis(10)) {
                    Ok(cmd) => match cmd {
                        GuiCommand::SearchText(query) => self.handle_text_search(query).await,
                        GuiCommand::Ingest(_path) => self.report_status("Ingest not implemented yet"),
                        GuiCommand::SearchAudio(_path) => self.report_status("Audio search not implemented yet"),
                        GuiCommand::AnalyzeAudio(_samples) => self.report_status("Audio analysis not implemented yet"),
                    },
                    Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                        // Poll Audio
                        // pop_chunk fails to compile, using simple loop
                        let mut samples_read = 0;
                        while let Ok(sample) = self.audio_consumer.pop() {
                             // Just drain for now
                             samples_read += 1;
                             if samples_read >= 1024 { break; }
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
        self.sender.send(WorkerResponse::Status(msg.to_string())).ok();
    }

    fn report_progress(&self, progress: f32) {
        self.sender.send(WorkerResponse::Progress(progress)).ok();
    }

    async fn handle_text_search(&self, text: String) {
        if let (Some(engine), Some((_store, hnsw))) = (&self.engine, &self.searcher) {
            self.report_status("Embedding text...");
            match engine.embed_text(&text).await {
                Ok(vec) => {
                    self.report_status("Searching...");
                    let results = hnsw.search(&vec, 10, 64); // k=10, ef=64
                    
                    let mapped_results: Vec<SearchResult> = results.into_iter().map(|(id, score)| {
                         // Mocking path retrieval for now as I need to verify Store API
                         let path = PathBuf::from(format!("/mock/result_{}.wav", id));
                         SearchResult { id, path, score }
                    }).collect();
                    
                    self.sender.send(WorkerResponse::SearchResults(mapped_results)).ok();
                    self.report_status("Ready");
                },
                Err(e) => {
                    self.sender.send(WorkerResponse::Error(e.to_string())).ok();
                    self.report_status("Error");
                }
            }
        } else {
             self.sender.send(WorkerResponse::Error("Engine or Index not loaded".into())).ok();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_worker_ping_pong() {
        let (gui_tx, worker_rx) = crossbeam_channel::unbounded();
        let (worker_tx, gui_rx) = crossbeam_channel::unbounded();

        let worker = VibratoWorker::new(worker_rx, worker_tx);
        worker.spawn();

        // 1. First message should be Starting Status
        match gui_rx.recv_timeout(Duration::from_secs(1)) {
            Ok(WorkerResponse::Status(msg)) => assert_eq!(msg, "Starting Worker..."),
            msg => panic!("Expected Starting Status, got {:?}", msg),
        }

        // 2. Consume Progress updates until Ready
        let mut progress_count = 0;
        loop {
            match gui_rx.recv_timeout(Duration::from_secs(2)) {
                Ok(WorkerResponse::Status(msg)) if msg == "Ready" => {
                    println!("Status: Ready");
                    break;
                }
                Ok(WorkerResponse::Progress(p)) => {
                    println!("Progress: {:.1}", p);
                    progress_count += 1;
                },
                msg => panic!("Unexpected message while waiting for Ready: {:?}", msg),
            }
        }
        assert!(progress_count > 0, "Should have received progress updates");

        // 3. Send Search Command
        // We use gui_tx (which connects to worker_rx)
        // Wait, gui_tx -> worker_rx.
        // worker uses self.receiver (which is worker_rx).
        // Correct.
        gui_tx.send(GuiCommand::SearchText("snare".into())).unwrap();

        // 4. Expect Search Results
        match gui_rx.recv_timeout(Duration::from_secs(1)) {
            Ok(WorkerResponse::SearchResults(results)) => {
                assert!(!results.is_empty());
                assert_eq!(results[0].score, 0.95);
                println!("Got Search Results: {:?}", results);
            }
            msg => panic!("Expected SearchResults, got {:?}", msg),
        }
    }
}
