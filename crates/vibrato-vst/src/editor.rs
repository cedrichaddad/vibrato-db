use nih_plug::prelude::*;
use nih_plug_egui::create_egui_editor;
use std::sync::Arc;
use parking_lot::RwLock;
use crossbeam_channel::Sender;
use crate::commands::{GuiCommand, WorkerResponse, SearchResult};
use crossbeam_channel::Receiver;
use crate::VibratoParams;

pub fn create_editor(
    params: Arc<VibratoParams>,
    current_results: Arc<RwLock<Vec<SearchResult>>>,
    status_msg: Arc<RwLock<String>>,
    progress: Arc<RwLock<f32>>,
    job_sender: Sender<GuiCommand>,
    result_receiver: Receiver<WorkerResponse>,
) -> Option<Box<dyn Editor>> {
    create_egui_editor(
        params.editor_state.clone(),
        (),
        |_, _| {},
        move |egui_ctx, _setter, _state| {
            // 1. Poll Worker Updates
            while let Ok(msg) = result_receiver.try_recv() {
                match msg {
                    WorkerResponse::Status(s) => *status_msg.write() = s,
                    WorkerResponse::Progress(p) => *progress.write() = p,
                    WorkerResponse::SearchResults(r) => *current_results.write() = r,
                    WorkerResponse::Error(e) => *status_msg.write() = format!("Error: {}", e),
                    _ => {}
                }
            }

            egui::CentralPanel::default().show(egui_ctx, |ui: &mut egui::Ui| {
                ui.heading("Vibrato AI");
                
                // Status Bar
                let status = status_msg.read().clone();
                ui.label(format!("Status: {}", status));

                // Progress Bar
                let prog = *progress.read();
                if prog < 1.0 {
                    ui.add(egui::ProgressBar::new(prog).show_percentage());
                }

                // Search Bar
                ui.horizontal(|ui| {
                     // We need a place to store the search text. 
                     // Since this closure is called every frame, we can't store it here easily 
                     // unless we use egui's memory or move a mutable string into the closure.
                     // But the closure is `FnMut`, so we can capture a RefCell or similar?
                     // Or just use a static/global for now, or use `ui.data()`
                     // A simple hack for now: Use `egui::TextEdit` with a persistent ID string?
                     // "SearchText" is the ID.
                     let mut search_text = ui.memory(|mem| mem.data.get_temp::<String>(egui::Id::new("search_text"))).unwrap_or_default();
                     
                     if ui.text_edit_singleline(&mut search_text).lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                         job_sender.send(GuiCommand::SearchText(search_text.clone())).ok();
                     }
                     
                     if ui.button("Search").clicked() {
                         job_sender.send(GuiCommand::SearchText(search_text.clone())).ok();
                     }
                     
                     ui.memory_mut(|mem| mem.data.insert_temp(egui::Id::new("search_text"), search_text));
                });
                
                ui.separator();

                // Results List
                let results = current_results.read();
                egui::ScrollArea::vertical().show(ui, |ui| {
                    if results.is_empty() {
                        ui.label("No results found.");
                    } else {
                        for result in results.iter() {
                            ui.group(|ui| {
                                let label = ui.label(format!("Score: {:.2}", result.score));
                                // Ghost Dragging implementation
                                // We use dnd_drag_source with the path as payload
                                if let Some(path_str) = result.path.to_str() {
                                    let source_id = egui::Id::new(result.id);
                                    let payload = path_str.to_string();
                                    
                                    ui.dnd_drag_source(source_id, payload, |ui| {
                                        // The draggable item
                                        if ui.button(format!("{:?}", result.path.file_name().unwrap_or_default()))
                                            .on_hover_text("Drag to DAW")
                                            .clicked() 
                                        {
                                            // Click handler (e.g. preview)
                                        }
                                        ui.label(format!("Dragging {:?}", result.path.file_name().unwrap_or_default()));
                                    });
                                }
                            });
                        }
                    }
                });
                // We will implement D&D here later
            });
        },
    )
}
