use std::path::PathBuf;
// use serde::{Serialize, Deserialize};

#[derive(Debug, Clone)]
pub enum GuiCommand {
    /// User typed "Funky Drum" and hit Enter
    SearchText(String),
    /// User dragged a file onto the "Search" zone
    SearchAudio(PathBuf),
    /// User dragged a folder to ingest
    Ingest(PathBuf),
    /// Audio query from ring buffer
    AnalyzeAudio(Vec<f32>),
}

#[derive(Debug, Clone)]
pub enum WorkerResponse {
    /// Search finished, here are the results
    SearchResults(Vec<SearchResult>),
    /// Index updated (after ingest)
    IndexUpdated { count: usize },
    /// Something broke (e.g. Model load failed)
    Error(String),
    /// Status update (e.g. "Loading Models... 50%")
    Status(String),
    /// Loading progress (0.0 to 1.0)
    Progress(f32),
    /// Unrecoverable error (worker crashed)
    FatalError(String),
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: usize,
    pub path: PathBuf, // We need the path to drag it back out!
    pub score: f32,
}
