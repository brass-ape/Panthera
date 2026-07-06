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

/// One Server-Sent Event frame from POST /api/chat/stream (see
/// webapp.py's chat_stream handler for the Python side of this
/// contract). Untagged/unrecognized `type` values are simply left
/// undeserializable by serde and dropped by parse_sse_frame below.
#[derive(Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
enum StreamEvent {
    Token { text: String },
    ToolCall {
        tool: String,
        #[serde(default)]
        #[allow(dead_code)] // not rendered yet, but part of the wire contract
        args: serde_json::Value,
    },
    ToolResult {
        #[allow(dead_code)]
        tool: String,
    },
    Final { text: String },
    Error { text: String },
}

/// Accumulates raw bytes from `ehttp::streaming::fetch` into complete
/// `\n\n`-delimited SSE frames, and holds the decoded events a frame
/// yields until the next `update()` drains them. Needs a mutex because
/// the streaming callback fires repeatedly from a background thread
/// (unlike the one-shot `Shared<T>` callbacks elsewhere in this file).
#[derive(Default)]
struct StreamState {
    byte_buffer: Vec<u8>,
    events: Vec<StreamEvent>,
}

fn parse_sse_frame(frame: &[u8]) -> Option<StreamEvent> {
    let text = std::str::from_utf8(frame).ok()?;
    for line in text.lines() {
        if let Some(json_str) = line.strip_prefix("data: ") {
            return serde_json::from_str(json_str).ok();
        }
    }
    None
}

/// Drains complete `\n\n`-terminated frames out of `state.byte_buffer`
/// into `state.events`, leaving any trailing partial frame in the
/// buffer for the next chunk to complete.
fn drain_sse_frames(state: &mut StreamState) {
    loop {
        let boundary = state
            .byte_buffer
            .windows(2)
            .position(|w| w == b"\n\n");
        let Some(pos) = boundary else { break };
        let frame: Vec<u8> = state.byte_buffer.drain(..pos + 2).collect();
        if let Some(event) = parse_sse_frame(&frame) {
            state.events.push(event);
        }
    }
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
    pending_stream: Arc<Mutex<StreamState>>,
    // Live, plain-text preview of the current reply while it streams
    // in -- same idea as main.py's rich.Live preview and the web UI's
    // "live" bubble: only the finished answer gets markdown-rendered.
    streaming_preview: Option<String>,
    streaming_status_lines: Vec<String>,
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
            pending_stream: Arc::new(Mutex::new(StreamState::default())),
            streaming_preview: None,
            streaming_status_lines: Vec::new(),
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
        self.streaming_preview = Some(String::new());
        self.streaming_status_lines.clear();

        let url = format!("{}/api/chat/stream", self.base_url);
        let body = serde_json::to_vec(&ChatRequestBody { message: &text })
            .expect("ChatRequestBody always serializes");
        let mut request = ehttp::Request::post(url, body);
        request.headers = ehttp::Headers::new(&[
            ("Accept", "text/event-stream"),
            ("Content-Type", "application/json"),
        ]);

        // Fresh shared state per request -- self.pending_stream is
        // swapped to this new Arc so update() only ever drains events
        // belonging to the turn currently in flight.
        let shared: Arc<Mutex<StreamState>> = Arc::new(Mutex::new(StreamState::default()));
        self.pending_stream = shared.clone();
        let ctx = ctx.clone();

        ehttp::streaming::fetch(request, move |result| {
            use std::ops::ControlFlow;
            match result {
                Ok(ehttp::streaming::Part::Response(response)) => {
                    if !response.ok {
                        let mut state = shared.lock().unwrap();
                        state.events.push(StreamEvent::Error {
                            text: format!("HTTP {}", response.status),
                        });
                        drop(state);
                        ctx.request_repaint();
                        return ControlFlow::Break(());
                    }
                    ControlFlow::Continue(())
                }
                Ok(ehttp::streaming::Part::Chunk(chunk)) => {
                    if chunk.is_empty() {
                        // End of stream (see ehttp::streaming::Part::Chunk's
                        // docs) -- nothing left to read.
                        return ControlFlow::Break(());
                    }
                    let mut state = shared.lock().unwrap();
                    state.byte_buffer.extend_from_slice(&chunk);
                    drain_sse_frames(&mut state);
                    drop(state);
                    ctx.request_repaint();
                    ControlFlow::Continue(())
                }
                Err(err) => {
                    let mut state = shared.lock().unwrap();
                    state.events.push(StreamEvent::Error {
                        text: format!("Could not reach the assistant server: {err}"),
                    });
                    drop(state);
                    ctx.request_repaint();
                    ControlFlow::Break(())
                }
            }
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

        let stream_events: Vec<StreamEvent> = {
            let mut state = self.pending_stream.lock().unwrap();
            state.events.drain(..).collect()
        };
        for event in stream_events {
            match event {
                StreamEvent::Token { text } => {
                    if let Some(preview) = &mut self.streaming_preview {
                        preview.push_str(&text);
                    }
                }
                StreamEvent::ToolCall { tool, .. } => {
                    self.streaming_status_lines.push(format!("→ using tool: {tool}"));
                    // Next iteration's tokens start a fresh preview,
                    // same as the CLI/web UI.
                    self.streaming_preview = Some(String::new());
                }
                StreamEvent::ToolResult { .. } => {}
                StreamEvent::Final { text } => {
                    self.pending = false;
                    self.streaming_preview = None;
                    self.streaming_status_lines.clear();
                    self.messages.push(ChatMessage {
                        role: Role::Assistant,
                        text,
                    });
                }
                StreamEvent::Error { text } => {
                    self.pending = false;
                    self.streaming_preview = None;
                    self.streaming_status_lines.clear();
                    self.messages.push(ChatMessage {
                        role: Role::Error,
                        text,
                    });
                }
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
                            for line in &self.streaming_status_lines {
                                ui.label(RichText::new(line.as_str()).weak().color(theme::AQUA));
                            }
                            if let Some(preview) = &self.streaming_preview {
                                if !preview.is_empty() {
                                    render_live_preview_bubble(ui, preview);
                                    ui.add_space(8.0);
                                }
                            }
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
                render_markdown(ui, &message.text, text_color);
            });
    });
}

/// The bubble shown while a reply is still streaming in -- plain text
/// only (`white-space: pre-wrap`-equivalent via egui's default label
/// wrapping), never markdown-rendered. Markdown formatting only makes
/// sense once the full answer is known (a "##" or "**" at the very
/// end of the visible stream so far might just not have its closing
/// half yet), matching the CLI and web UI's same "raw while streaming,
/// formatted once final" behavior.
fn render_live_preview_bubble(ui: &mut egui::Ui, text: &str) {
    let layout = egui::Layout::left_to_right(egui::Align::TOP);
    ui.with_layout(layout, |ui| {
        let max_width = (ui.available_width() * 0.78).max(120.0);
        egui::Frame::new()
            .fill(theme::VIOLET.gamma_multiply(0.55))
            .corner_radius(CornerRadius::same(14))
            .inner_margin(Margin::symmetric(12, 9))
            .show(ui, |ui| {
                ui.set_max_width(max_width);
                ui.label(RichText::new(text).color(theme::TEXT));
            });
    });
}

/// Splits a heading's leading "#" run off, e.g. "## Title" -> Some((2, "Title")).
fn heading_prefix(line: &str) -> Option<(usize, &str)> {
    let hashes = line.bytes().take_while(|&b| b == b'#').count();
    if hashes == 0 || hashes > 6 {
        return None;
    }
    line[hashes..].strip_prefix(' ').map(|rest| (hashes, rest))
}

/// Splits a numbered-list line's leading "N. " off, e.g.
/// "2. Second point" -> Some(("2", "Second point")).
fn numbered_prefix(line: &str) -> Option<(&str, &str)> {
    let (head, rest) = line.split_once(". ")?;
    if head.is_empty() || !head.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    Some((head, rest))
}

/// Builds a single-line `LayoutJob` handling `` `code` `` spans and
/// `**bold**` emphasis (rendered in the accent color rather than a
/// true bold font weight, since egui's default fonts don't ship a
/// bold variant without embedding one) -- a deliberately small subset
/// of inline markdown, matching the web UI's app.js renderer.
fn inline_layout_job(line: &str, color: Color32, body_size: f32) -> egui::text::LayoutJob {
    let mut job = egui::text::LayoutJob::default();
    let font_id = egui::FontId::proportional(body_size);
    let mono_id = egui::FontId::monospace(body_size * 0.95);

    for (i, code_part) in line.split('`').enumerate() {
        if code_part.is_empty() {
            continue;
        }
        if i % 2 == 1 {
            job.append(
                code_part,
                0.0,
                egui::TextFormat {
                    font_id: mono_id.clone(),
                    color,
                    background: Color32::from_black_alpha(60),
                    ..Default::default()
                },
            );
        } else {
            for (j, bold_part) in code_part.split("**").enumerate() {
                if bold_part.is_empty() {
                    continue;
                }
                let part_color = if j % 2 == 1 { theme::PEACH } else { color };
                job.append(
                    bold_part,
                    0.0,
                    egui::TextFormat {
                        font_id: font_id.clone(),
                        color: part_color,
                        ..Default::default()
                    },
                );
            }
        }
    }
    job
}

/// Small line-oriented markdown renderer: headers, bullet/numbered
/// lists, paragraphs, plus the inline `**bold**`/`` `code` `` handling
/// from inline_layout_job. Mirrors web/static/js/app.js's renderContent
/// -- egui has no HTML/commonmark renderer built in, so this covers
/// the same deliberately small subset by hand rather than pulling in
/// a markdown crate for one chat view.
fn render_markdown(ui: &mut egui::Ui, text: &str, color: Color32) {
    let body_size = egui::TextStyle::Body.resolve(ui.style()).size;

    for raw_line in text.split('\n') {
        let line = raw_line.trim();
        if line.is_empty() {
            ui.add_space(4.0);
            continue;
        }

        if let Some((level, content)) = heading_prefix(line) {
            let size = match level {
                1 => body_size + 3.0,
                2 => body_size + 2.0,
                _ => body_size + 1.0,
            };
            ui.add_space(2.0);
            ui.label(inline_layout_job(content, theme::PEACH, size));
            continue;
        }

        if let Some(rest) = line.strip_prefix("- ").or_else(|| line.strip_prefix("* ")) {
            ui.horizontal(|ui| {
                ui.add_space(4.0);
                ui.colored_label(theme::AQUA, "•");
                ui.label(inline_layout_job(rest, color, body_size));
            });
            continue;
        }

        if let Some((number, rest)) = numbered_prefix(line) {
            ui.horizontal(|ui| {
                ui.add_space(4.0);
                ui.colored_label(theme::AQUA, format!("{number}."));
                ui.label(inline_layout_job(rest, color, body_size));
            });
            continue;
        }

        ui.label(inline_layout_job(line, color, body_size));
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn events_from_bytes(chunks: &[&[u8]]) -> Vec<StreamEvent> {
        let mut state = StreamState::default();
        for chunk in chunks {
            state.byte_buffer.extend_from_slice(chunk);
            drain_sse_frames(&mut state);
        }
        state.events
    }

    fn token_text(event: &StreamEvent) -> &str {
        match event {
            StreamEvent::Token { text } => text,
            _ => panic!("expected a Token event"),
        }
    }

    #[test]
    fn parses_a_single_complete_frame() {
        let events = events_from_bytes(&[br#"data: {"type": "token", "text": "hi"}

"#]);
        assert_eq!(events.len(), 1);
        assert_eq!(token_text(&events[0]), "hi");
    }

    #[test]
    fn parses_multiple_frames_in_one_chunk() {
        let events = events_from_bytes(&[br#"data: {"type": "token", "text": "a"}

data: {"type": "token", "text": "b"}

"#]);
        assert_eq!(events.len(), 2);
        assert_eq!(token_text(&events[0]), "a");
        assert_eq!(token_text(&events[1]), "b");
    }

    #[test]
    fn reassembles_a_frame_split_across_chunks() {
        // The frame's own JSON, and even the "\n\n" terminator, can
        // land on either side of a network read boundary -- this is
        // the whole reason StreamState buffers bytes instead of
        // parsing each `Part::Chunk` in isolation.
        let events = events_from_bytes(&[
            br#"data: {"type": "tok"#,
            br#"en", "text": "hello"}

"#,
        ]);
        assert_eq!(events.len(), 1);
        assert_eq!(token_text(&events[0]), "hello");
    }

    #[test]
    fn reassembles_when_the_double_newline_itself_is_split() {
        let events = events_from_bytes(&[
            br#"data: {"type": "token", "text": "x"}
"#,
            b"\n",
        ]);
        assert_eq!(events.len(), 1);
        assert_eq!(token_text(&events[0]), "x");
    }

    #[test]
    fn leaves_an_incomplete_trailing_frame_buffered() {
        let mut state = StreamState::default();
        state
            .byte_buffer
            .extend_from_slice(br#"data: {"type": "token", "text": "partial""#);
        drain_sse_frames(&mut state);
        assert!(state.events.is_empty());
        assert!(!state.byte_buffer.is_empty());
    }

    #[test]
    fn parses_tool_call_event_and_ignores_unknown_extra_fields() {
        let events = events_from_bytes(&[
            br#"data: {"type": "tool_call", "tool": "web_search", "args": {"query": "cats"}}

"#,
        ]);
        assert_eq!(events.len(), 1);
        match &events[0] {
            StreamEvent::ToolCall { tool, .. } => assert_eq!(tool, "web_search"),
            _ => panic!("expected a ToolCall event"),
        }
    }

    #[test]
    fn parses_final_and_error_events() {
        let final_events = events_from_bytes(&[br#"data: {"type": "final", "text": "done"}

"#]);
        match &final_events[0] {
            StreamEvent::Final { text } => assert_eq!(text, "done"),
            _ => panic!("expected a Final event"),
        }

        let error_events = events_from_bytes(&[br#"data: {"type": "error", "text": "oops"}

"#]);
        match &error_events[0] {
            StreamEvent::Error { text } => assert_eq!(text, "oops"),
            _ => panic!("expected an Error event"),
        }
    }

    #[test]
    fn malformed_frame_is_skipped_not_fatal() {
        let events = events_from_bytes(&[b"data: not valid json\n\n"]);
        assert!(events.is_empty());
    }

    #[test]
    fn value_to_edit_string_unwraps_json_strings() {
        assert_eq!(
            value_to_edit_string(&serde_json::Value::String("hello".to_string())),
            "hello"
        );
        assert_eq!(value_to_edit_string(&serde_json::json!(42)), "42");
        assert_eq!(value_to_edit_string(&serde_json::json!(0.5)), "0.5");
    }

    #[test]
    fn heading_prefix_parses_level_and_content() {
        assert_eq!(heading_prefix("## Size and Weight"), Some((2, "Size and Weight")));
        assert_eq!(heading_prefix("# Title"), Some((1, "Title")));
        assert_eq!(heading_prefix("Not a heading"), None);
        assert_eq!(heading_prefix("#NoSpace"), None);
    }

    #[test]
    fn numbered_prefix_parses_number_and_content() {
        assert_eq!(numbered_prefix("1. First point"), Some(("1", "First point")));
        assert_eq!(numbered_prefix("12. Twelfth"), Some(("12", "Twelfth")));
        assert_eq!(numbered_prefix("Not numbered"), None);
        assert_eq!(numbered_prefix("1.NoSpace"), None);
    }
}
