//! Desktop GUI for the local AI assistant.
//!
//! This talks to `webapp.py` (the Flask front-end also used by the
//! browser UI) over HTTP -- it holds no agent/tool-calling logic of
//! its own, matching the "GUI is just another front-end that calls
//! Agent.run_turn" architecture described in claude.md. Start
//! `webapp.py` first, then run this.

use std::sync::{Arc, Mutex};

use eframe::egui;
use egui::{Color32, CornerRadius, Margin, RichText, Stroke};
use serde::{Deserialize, Serialize};

mod theme {
    use egui::Color32;

    pub const NAVY: Color32 = Color32::from_rgb(0x17, 0x17, 0x38);
    pub const NAVY_LIGHT: Color32 = Color32::from_rgb(0x22, 0x20, 0x54);
    pub const CORAL: Color32 = Color32::from_rgb(0xdb, 0x54, 0x61);
    pub const PEACH: Color32 = Color32::from_rgb(0xff, 0xd9, 0xce);
    pub const VIOLET: Color32 = Color32::from_rgb(0x59, 0x3c, 0x8f);
    pub const AQUA: Color32 = Color32::from_rgb(0x8e, 0xf9, 0xf3);
    pub const TEXT: Color32 = Color32::from_rgb(0xf5, 0xf3, 0xff);
}

#[derive(Clone, Copy, PartialEq)]
enum Role {
    User,
    Assistant,
    Error,
}

#[derive(Clone)]
struct ChatMessage {
    role: Role,
    text: String,
}

#[derive(Serialize)]
struct ChatRequestBody<'a> {
    message: &'a str,
}

#[derive(Deserialize, Default)]
struct ChatResponseBody {
    reply: Option<String>,
    error: Option<String>,
}

#[derive(Deserialize, Clone, Default)]
struct StatusInfo {
    backend: String,
    model: String,
    web_search_backend: String,
}

/// Shape of GET/POST /api/config's response -- `fields` maps setting
/// name to its type ("str"/"int"/"float"), `values` holds the current
/// value of each (mixed types, hence serde_json::Value), and
/// `overridden` lists which ones are explicitly set in config.json.
#[derive(Deserialize, Clone, Default)]
struct ConfigInfo {
    fields: std::collections::BTreeMap<String, String>,
    values: std::collections::BTreeMap<String, serde_json::Value>,
    overridden: Vec<String>,
}

#[derive(Deserialize, Default)]
struct ConfigErrorBody {
    error: Option<String>,
}

fn value_to_edit_string(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Shared slot a background `ehttp` callback writes into; the next
/// `update()` call drains it. `ehttp::fetch`'s callback runs on
/// whatever thread the platform's HTTP stack uses, never the UI
/// thread, so state can't be mutated directly from it -- this is the
/// standard eframe+ehttp handoff pattern.
type Shared<T> = Arc<Mutex<Option<T>>>;

struct AssistantApp {
    base_url: String,
    messages: Vec<ChatMessage>,
    input: String,
    pending: bool,
    pending_reply: Shared<Result<String, String>>,
    status: Option<StatusInfo>,
    pending_status: Shared<Result<StatusInfo, String>>,

    settings_open: bool,
    settings_info: Option<ConfigInfo>,
    settings_edits: std::collections::BTreeMap<String, String>,
    settings_error: Option<String>,
    settings_saving: bool,
    pending_config: Shared<Result<ConfigInfo, String>>,
    pending_save: Shared<Result<ConfigInfo, String>>,
}

impl AssistantApp {
    fn new(cc: &eframe::CreationContext<'_>, base_url: String) -> Self {
        configure_visuals(&cc.egui_ctx);

        let app = Self {
            base_url,
            messages: vec![ChatMessage {
                role: Role::Assistant,
                text: "Hi! Ask me anything -- I can remember things across \
                       conversations and, if enabled, search the web."
                    .to_string(),
            }],
            input: String::new(),
            pending: false,
            pending_reply: Arc::new(Mutex::new(None)),
            status: None,
            pending_status: Arc::new(Mutex::new(None)),

            settings_open: false,
            settings_info: None,
            settings_edits: std::collections::BTreeMap::new(),
            settings_error: None,
            settings_saving: false,
            pending_config: Arc::new(Mutex::new(None)),
            pending_save: Arc::new(Mutex::new(None)),
        };
        app.fetch_status(cc.egui_ctx.clone());
        app
    }

    fn fetch_status(&self, ctx: egui::Context) {
        let url = format!("{}/api/status", self.base_url);
        let shared = self.pending_status.clone();
        ehttp::fetch(ehttp::Request::get(url), move |result| {
            let parsed = match result {
                Ok(response) if response.ok => response
                    .json::<StatusInfo>()
                    .map_err(|err| err.to_string()),
                Ok(response) => Err(format!("HTTP {}", response.status)),
                Err(err) => Err(err),
            };
            *shared.lock().unwrap() = Some(parsed);
            ctx.request_repaint();
        });
    }

    fn send(&mut self, ctx: &egui::Context) {
        let text = self.input.trim().to_string();
        if text.is_empty() || self.pending {
            return;
        }
        self.messages.push(ChatMessage {
            role: Role::User,
            text: text.clone(),
        });
        self.input.clear();
        self.pending = true;

        let url = format!("{}/api/chat", self.base_url);
        let body = serde_json::to_vec(&ChatRequestBody { message: &text })
            .expect("ChatRequestBody always serializes");
        let mut request = ehttp::Request::post(url, body);
        request.headers = ehttp::Headers::new(&[
            ("Accept", "application/json"),
            ("Content-Type", "application/json"),
        ]);

        let shared = self.pending_reply.clone();
        let ctx = ctx.clone();
        ehttp::fetch(request, move |result| {
            let parsed: Result<String, String> = match result {
                Ok(response) => {
                    let body: ChatResponseBody = response.json().unwrap_or_default();
                    if let Some(reply) = body.reply {
                        Ok(reply)
                    } else if let Some(error) = body.error {
                        Err(error)
                    } else {
                        Err(format!("Unexpected response (HTTP {})", response.status))
                    }
                }
                Err(err) => Err(format!("Could not reach the assistant server: {err}")),
            };
            *shared.lock().unwrap() = Some(parsed);
            ctx.request_repaint();
        });
    }

    fn fetch_config(&self, ctx: egui::Context) {
        let url = format!("{}/api/config", self.base_url);
        let shared = self.pending_config.clone();
        ehttp::fetch(ehttp::Request::get(url), move |result| {
            let parsed = match result {
                Ok(response) if response.ok => {
                    response.json::<ConfigInfo>().map_err(|err| err.to_string())
                }
                Ok(response) => Err(format!("HTTP {}", response.status)),
                Err(err) => Err(err),
            };
            *shared.lock().unwrap() = Some(parsed);
            ctx.request_repaint();
        });
    }

    fn save_config(&mut self, ctx: &egui::Context) {
        let url = format!("{}/api/config", self.base_url);
        let body = serde_json::to_vec(&self.settings_edits).expect("settings_edits always serializes");
        let mut request = ehttp::Request::post(url, body);
        request.headers = ehttp::Headers::new(&[
            ("Accept", "application/json"),
            ("Content-Type", "application/json"),
        ]);

        self.settings_saving = true;
        let shared = self.pending_save.clone();
        let ctx = ctx.clone();
        ehttp::fetch(request, move |result| {
            let parsed = match result {
                Ok(response) if response.ok => {
                    response.json::<ConfigInfo>().map_err(|err| err.to_string())
                }
                Ok(response) => {
                    let message = response
                        .json::<ConfigErrorBody>()
                        .ok()
                        .and_then(|body| body.error)
                        .unwrap_or_else(|| format!("HTTP {}", response.status));
                    Err(message)
                }
                Err(err) => Err(format!("Could not reach the assistant server: {err}")),
            };
            *shared.lock().unwrap() = Some(parsed);
            ctx.request_repaint();
        });
    }

    fn show_settings_window(&mut self, ctx: &egui::Context) {
        let mut open = self.settings_open;
        egui::Window::new("Settings")
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .frame(
                egui::Frame::new()
                    .fill(theme::NAVY_LIGHT)
                    .stroke(Stroke::new(1.0, theme::VIOLET))
                    .corner_radius(CornerRadius::same(16))
                    .inner_margin(Margin::same(16)),
            )
            .show(ctx, |ui| {
                self.render_settings_contents(ui, ctx);
            });
        self.settings_open = open;
    }

    fn render_settings_contents(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.set_width(320.0);
        ui.label(
            RichText::new("Changes save to config.json and take effect immediately.")
                .color(theme::TEXT)
                .weak(),
        );
        ui.add_space(8.0);

        if let Some(error) = self.settings_error.clone() {
            ui.colored_label(theme::CORAL, error);
            ui.add_space(6.0);
        }

        match self.settings_info.clone() {
            None => {
                ui.horizontal(|ui| {
                    ui.add(egui::Spinner::new().color(theme::AQUA));
                    ui.label(RichText::new("Loading…").weak());
                });
            }
            Some(info) => {
                let overridden: std::collections::HashSet<String> =
                    info.overridden.iter().cloned().collect();
                egui::ScrollArea::vertical()
                    .max_height(360.0)
                    .show(ui, |ui| {
                        for name in info.fields.keys() {
                            let mut label = name.replace('_', " ");
                            if overridden.contains(name) {
                                label.push_str("  •");
                            }
                            ui.label(RichText::new(label).color(theme::TEXT).small());
                            if let Some(value) = self.settings_edits.get_mut(name) {
                                ui.add(egui::TextEdit::singleline(value).desired_width(f32::INFINITY));
                            }
                            ui.add_space(6.0);
                        }
                    });
            }
        }

        ui.add_space(6.0);
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let button = egui::Button::new(RichText::new("Save").color(theme::NAVY).strong())
                .fill(theme::CORAL)
                .corner_radius(CornerRadius::same(8));
            let enabled = !self.settings_saving && self.settings_info.is_some();
            if ui.add_enabled(enabled, button).clicked() {
                self.save_config(ctx);
            }
            if self.settings_saving {
                ui.add(egui::Spinner::new().color(theme::AQUA));
            }
        });
    }
}

fn configure_visuals(ctx: &egui::Context) {
    let mut visuals = egui::Visuals::dark();
    visuals.panel_fill = theme::NAVY;
    visuals.window_fill = theme::NAVY_LIGHT;
    visuals.extreme_bg_color = theme::NAVY;
    visuals.faint_bg_color = theme::NAVY_LIGHT;
    visuals.selection.bg_fill = theme::VIOLET;
    visuals.selection.stroke = Stroke::new(1.0, theme::AQUA);
    visuals.hyperlink_color = theme::AQUA;
    visuals.widgets.inactive.bg_fill = theme::NAVY_LIGHT;
    visuals.widgets.hovered.bg_fill = theme::VIOLET.gamma_multiply(0.8);
    visuals.widgets.active.bg_fill = theme::VIOLET;
    visuals.override_text_color = Some(theme::TEXT);
    ctx.set_visuals(visuals);
}

impl eframe::App for AssistantApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if let Some(result) = self.pending_status.lock().unwrap().take() {
            if let Ok(status) = result {
                self.status = Some(status);
            }
        }

        if let Some(result) = self.pending_reply.lock().unwrap().take() {
            self.pending = false;
            match result {
                Ok(reply) => self.messages.push(ChatMessage {
                    role: Role::Assistant,
                    text: reply,
                }),
                Err(err) => self.messages.push(ChatMessage {
                    role: Role::Error,
                    text: err,
                }),
            }
        }

        if let Some(result) = self.pending_config.lock().unwrap().take() {
            match result {
                Ok(info) => {
                    self.settings_edits = info
                        .values
                        .iter()
                        .map(|(k, v)| (k.clone(), value_to_edit_string(v)))
                        .collect();
                    self.settings_info = Some(info);
                    self.settings_error = None;
                }
                Err(err) => self.settings_error = Some(err),
            }
        }

        if let Some(result) = self.pending_save.lock().unwrap().take() {
            self.settings_saving = false;
            match result {
                Ok(info) => {
                    self.settings_edits = info
                        .values
                        .iter()
                        .map(|(k, v)| (k.clone(), value_to_edit_string(v)))
                        .collect();
                    self.settings_info = Some(info);
                    self.settings_error = None;
                    self.settings_open = false;
                    // Backend/model may have changed -- refresh the status pill.
                    self.fetch_status(ctx.clone());
                }
                Err(err) => self.settings_error = Some(err),
            }
        }

        egui::TopBottomPanel::top("topbar")
            .frame(egui::Frame::new().fill(theme::NAVY).inner_margin(Margin::symmetric(14, 10)))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new("●").color(theme::AQUA).size(14.0));
                    ui.label(
                        RichText::new("Local AI Assistant")
                            .strong()
                            .size(17.0)
                            .color(theme::PEACH),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let gear = egui::Button::new(RichText::new("⚙").color(theme::TEXT))
                            .fill(Color32::TRANSPARENT);
                        if ui.add(gear).on_hover_text("Settings").clicked() {
                            self.settings_open = true;
                            self.fetch_config(ctx.clone());
                        }
                        ui.separator();
                        match &self.status {
                            Some(status) => {
                                ui.label(
                                    RichText::new(format!("{} · {}", status.backend, status.model))
                                        .color(theme::TEXT)
                                        .weak(),
                                );
                                ui.separator();
                                ui.label(
                                    RichText::new(format!("web search: {}", status.web_search_backend))
                                        .color(theme::TEXT)
                                        .weak(),
                                );
                            }
                            None => {
                                ui.label(RichText::new("connecting…").weak());
                            }
                        }
                    });
                });
            });

        if self.settings_open {
            self.show_settings_window(ctx);
        }

        egui::TopBottomPanel::bottom("composer")
            .frame(egui::Frame::new().fill(theme::NAVY).inner_margin(Margin::symmetric(14, 12)))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    let available = ui.available_width() - 56.0;
                    let response = ui.add_sized(
                        [available.max(80.0), 34.0],
                        egui::TextEdit::singleline(&mut self.input)
                            .hint_text("Type a message...")
                            .margin(Margin::symmetric(10, 8)),
                    );
                    let enter_pressed =
                        response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));

                    let button = egui::Button::new(RichText::new("➤").color(theme::NAVY).strong())
                        .fill(theme::CORAL)
                        .corner_radius(CornerRadius::same(17));
                    let clicked = ui
                        .add_sized([34.0, 34.0], button)
                        .on_hover_text("Send")
                        .clicked();

                    if (clicked || enter_pressed) && !self.pending {
                        self.send(ctx);
                        response.request_focus();
                    }
                });
            });

        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(theme::NAVY).inner_margin(Margin::symmetric(14, 10)))
            .show(ctx, |ui| {
                egui::ScrollArea::vertical()
                    .stick_to_bottom(true)
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        for message in &self.messages {
                            render_bubble(ui, message);
                            ui.add_space(8.0);
                        }
                        if self.pending {
                            ui.horizontal(|ui| {
                                ui.add(egui::Spinner::new().color(theme::AQUA));
                                ui.label(RichText::new("thinking…").weak());
                            });
                        }
                    });
            });
    }
}

fn render_bubble(ui: &mut egui::Ui, message: &ChatMessage) {
    let (fill, text_color, align_right) = match message.role {
        Role::User => (theme::AQUA, theme::NAVY, true),
        Role::Assistant => (theme::VIOLET.gamma_multiply(0.55), theme::TEXT, false),
        Role::Error => (theme::CORAL.gamma_multiply(0.75), Color32::WHITE, false),
    };

    let layout = if align_right {
        egui::Layout::right_to_left(egui::Align::TOP)
    } else {
        egui::Layout::left_to_right(egui::Align::TOP)
    };

    ui.with_layout(layout, |ui| {
        let max_width = (ui.available_width() * 0.78).max(120.0);
        egui::Frame::new()
            .fill(fill)
            .corner_radius(CornerRadius::same(14))
            .inner_margin(Margin::symmetric(12, 9))
            .show(ui, |ui| {
                ui.set_max_width(max_width);
                ui.label(RichText::new(&message.text).color(text_color));
            });
    });
}

fn main() -> eframe::Result {
    let base_url = std::env::var("ASSISTANT_GUI_BASE_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:5000".to_string());

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([460.0, 660.0])
            .with_min_inner_size([340.0, 420.0])
            .with_title("Local AI Assistant"),
        ..Default::default()
    };

    eframe::run_native(
        "Local AI Assistant",
        options,
        Box::new(move |cc| Ok(Box::new(AssistantApp::new(cc, base_url)))),
    )
}
