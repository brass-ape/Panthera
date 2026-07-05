# assistant-gui

A cross-platform desktop GUI for the local AI assistant, built with
[egui](https://github.com/emilk/egui)/[eframe](https://github.com/emilk/egui/tree/main/crates/eframe).

This holds no agent logic of its own -- it's purely a client for
`webapp.py`'s HTTP API (`/api/status`, `/api/chat`), the same way
`main.py`'s CLI and the browser UI are both just front-ends over
`Agent.run_turn`. That keeps the Python backend as the single source of
truth for tool calling, memory, and LLM communication, and this crate
free of any of that logic.

## Running

Start the Python backend first:

```bash
# from the repo root
pip install flask
python webapp.py
```

Then, in another terminal:

```bash
cd gui
cargo run --release
```

By default it connects to `http://127.0.0.1:5000`. Point it elsewhere
(e.g. a different machine on your network running `webapp.py`) with:

```bash
ASSISTANT_GUI_BASE_URL=http://192.168.1.50:5000 cargo run --release
```

## Cross-platform notes

`egui`/`eframe` are pure-Rust and windowing is handled by `winit`, so
the same code builds natively on Linux (X11 and Wayland, via
`winit`/`smithay-clipboard`/`sctk-adwaita`), Windows, and macOS with no
platform-specific code in this crate. On Linux Mint (X11 or Wayland
session) `cargo build --release` produces a single native binary with
no runtime dependency beyond the graphics stack (OpenGL via `glow`).

To build for another OS from Linux, use [`cross`](https://github.com/cross-rs/cross)
or build natively on that OS -- there's nothing in this crate that
needs conditional compilation to support it.

## Design

Same palette and "liquid glass" language as the browser UI
(`web/static/css/style.css`): deep navy background, aqua/violet/coral
accents, glass-toned message bubbles. egui is an immediate-mode,
GPU-rasterized UI, so it can't do CSS-style `backdrop-filter` blur --
the glass *look* here comes from translucent fills and rounded
corners rather than a literal blur, which is the closest equivalent
available in this toolkit.

## Status

A scrollable message list, a composer with Enter-to-send, a spinner
while waiting on a reply, a status strip showing the active
backend/model/web-search config (via `/api/status`), and a settings
panel (the ⚙ button) backed by `/api/config` -- edits save to
`config.json` on the Python side and take effect immediately, no
restart needed. No conversation persistence across restarts yet.
