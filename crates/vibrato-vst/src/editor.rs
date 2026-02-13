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
    search_text: Arc<RwLock<String>>,
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
                     let mut text = search_text.write();
                     
                     if ui.text_edit_singleline(&mut *text).lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                         job_sender.send(GuiCommand::SearchText(text.clone())).ok();
                     }
                     
                     if ui.button("Search").clicked() {
                         job_sender.send(GuiCommand::SearchText(text.clone())).ok();
                     }
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
                                let _label = ui.label(format!("Score: {:.2}", result.score));
                                // Ghost Dragging removed in favor of reliable Copy Path
                                if let Some(path_str) = result.path.to_str() {
                                    ui.horizontal(|ui| {
                                        ui.label(format!("{:?}", result.path.file_name().unwrap_or_default()));
                                        if ui.button("Copy Path").clicked() {
                                            ui.ctx().copy_text(path_str.to_string());
                                        }
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
