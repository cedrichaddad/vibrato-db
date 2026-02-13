use std::sync::Arc;
use parking_lot::RwLock;

use nih_plug::prelude::*;
use nih_plug_egui::{create_egui_editor, EguiState};

use crossbeam_channel::{Sender, Receiver};

mod commands;
mod editor;
mod worker;

use commands::{GuiCommand, WorkerResponse, SearchResult};
use worker::VibratoWorker;
use rtrb::{Producer, RingBuffer};

pub struct VibratoPlugin {
    params: Arc<VibratoParams>,
    
    // Communication channels
    job_sender: Sender<GuiCommand>,
    result_receiver: Receiver<WorkerResponse>,
    
    // Audio Capture
    audio_producer: Option<Producer<f32>>,
    
    // UI State (Cached for immediate rendering)
    current_results: Arc<RwLock<Vec<SearchResult>>>,
    status_msg: Arc<RwLock<String>>,
    progress: Arc<RwLock<f32>>,
}

#[derive(Params)]
pub(crate) struct VibratoParams {
    #[persist = "editor-state"]
    editor_state: Arc<EguiState>,
}

impl Default for VibratoParams {
    fn default() -> Self {
        Self {
            editor_state: EguiState::from_size(800, 600),
        }
    }
}

impl Default for VibratoPlugin {
    fn default() -> Self {
        let (tx, rx) = crossbeam_channel::unbounded();
        let (res_tx, res_rx) = crossbeam_channel::unbounded();
        
        // Default impl just creates dummy channels or holds them until init?
        // Ideally we don't spawn here.
        
        Self {
            params: Arc::new(VibratoParams::default()),
            job_sender: tx,
            result_receiver: res_rx,
            audio_producer: None,
            current_results: Arc::new(RwLock::new(Vec::new())),
            status_msg: Arc::new(RwLock::new("Initializing...".to_string())),
            progress: Arc::new(RwLock::new(0.0)),
        }
    }
}

impl Plugin for VibratoPlugin {
    const NAME: &'static str = "Vibrato AI";
    const VENDOR: &'static str = "Cedric Haddad";
    const URL: &'static str = "https://github.com/cedrichaddad/vibrato-db";
    const EMAIL: &'static str = "info@vibrato.ai";

    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    // IO ports
    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(2),
            main_output_channels: NonZeroU32::new(2),
            ..AudioIOLayout::const_default()
        },
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(1),
            main_output_channels: NonZeroU32::new(1),
            ..AudioIOLayout::const_default()
        },
    ];

    const MIDI_INPUT: MidiConfig = MidiConfig::None;
    const SAMPLE_ACCURATE_AUTOMATION: bool = true;

    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        _buffer_config: &BufferConfig,
        _context: &mut impl InitContext<Self>,
    ) -> bool {
        // Spawn the worker thread ONCE when plugin loads
        // We need to re-create channels here because default() created dummy ones?
        // Or if we created them in default(), we pass the OTHER ends to the worker.
        // The worker needs `Receiver<GuiCommand>` and `Sender<WorkerResponse>`.
        // But `default()` returns `VibratoPlugin` which holds `Sender<GuiCommand>` and `Receiver<WorkerResponse>`.
        // So we need to store the *other ends* temporarily? Or just recreate channels here.
        
        let (tx, rx) = crossbeam_channel::unbounded();
        let (res_tx, res_rx) = crossbeam_channel::unbounded();
        
        // Ring Buffer (e.g., 2 seconds of mono audio at 48kHz for analysis)
        // 48000 * 2 = 96000
        let (prod, cons) = RingBuffer::new(96000);
        
        let worker = VibratoWorker::new(rx, res_tx, cons);
        worker.spawn();
        
        self.job_sender = tx;
        self.result_receiver = res_rx;
        self.audio_producer = Some(prod);
        
        true
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        // AUDIO THREAD - KEEP EMPTY (Passthrough)
        // We strictly do NOT touch the vector DB here.
        
        // Simple passthrough since we are an instrument/effect but mostly a Librarian.
        // Actually, if we are an instrument we output silence? 
        // If effect, we pass input to output.
        // For now, let's just make it silent/pass-through.
        
        // Audio Capture for Search
        if let Some(producer) = &mut self.audio_producer {
             for channel in buffer.as_slice() {
                 // Interleave or just take first channel? 
                 // Simple approach: Mix down or take left channel.
                 // Let's take the first channel if available.
                 if !channel.is_empty() {
                     // Write chunk to ring buffer
                     // push_chunk fails to compile, using simple loop for scaffolding
                     for sample in channel.iter() {
                         if producer.is_full() { break; }
                         let _ = producer.push(*sample);
                     }
                     // Note: If we are stereo, we might be pushing only Left.
                     // If we overwrite, we might skip samples. 
                     // For analysis, one channel is usually enough for rhythm/timbre.
                     break; // Only capture first channel
                 }
             }
        }
        
        ProcessStatus::Normal
    }
    
    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        // Spawns the Egui window
        editor::create_editor(
            self.params.clone(),
            self.current_results.clone(),
            self.status_msg.clone(),
            self.progress.clone(),
            self.job_sender.clone(),
            self.result_receiver.clone(),
        )
    }
}

impl Vst3Plugin for VibratoPlugin {
    const VST3_CLASS_ID: [u8; 16] = *b"VibratoAIVectorD";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = &[
        Vst3SubCategory::Tools, 
        Vst3SubCategory::Analyzer
    ];
}

impl ClapPlugin for VibratoPlugin {
    const CLAP_ID: &'static str = "com.vibrato.ai";
    const CLAP_DESCRIPTION: Option<&'static str> = Some("Vibrato AI - Vector Database Plugin");
    const CLAP_MANUAL_URL: Option<&'static str> = Some("https://vibrato.ai/manual");
    const CLAP_SUPPORT_URL: Option<&'static str> = Some("https://vibrato.ai/support");
    const CLAP_FEATURES: &'static [ClapFeature] = &[ClapFeature::AudioEffect, ClapFeature::Utility];
}

nih_export_clap!(VibratoPlugin);
nih_export_vst3!(VibratoPlugin);
