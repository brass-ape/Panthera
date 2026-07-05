"""
main.py
=======

Command-line entry point for the local AI assistant.

Run with:

    python main.py

Type your message and press Enter. Type `exit` or `quit` (or press
Ctrl+C / Ctrl+D) to leave.
"""

from __future__ import annotations

import logging

from rich.console import Console
from rich.live import Live
from rich.logging import RichHandler
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from agent import Agent, LLMConnectionError, ToolCallEvent, ToolResultEvent, TokenEvent
from config import CONFIG

# A shared palette so the terminal UI and any future front-ends (see
# webapp.py) read as one visual system rather than two unrelated
# color schemes.
THEME = Theme(
    {
        "banner.border": "#8ef9f3",
        "banner.title": "bold #ffd9ce",
        "user.label": "bold #8ef9f3",
        "assistant.border": "#593c8f",
        "assistant.title": "bold #ffd9ce",
        "error": "bold #db5461",
        "muted": "dim #8ef9f3",
        "field": "#8ef9f3",
        "value": "bold #ffd9ce",
    }
)

console = Console(theme=THEME)


def configure_logging() -> None:
    """Set up logging to both a file and the console.

    Console only shows warnings and above by default so normal use
    isn't noisy; the log file captures everything at CONFIG.log_level
    for debugging.
    """
    root = logging.getLogger("assistant")
    root.setLevel(CONFIG.log_level)

    file_formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(CONFIG.log_path, encoding="utf-8")
    file_handler.setLevel(CONFIG.log_level)
    file_handler.setFormatter(file_formatter)

    console_handler = RichHandler(console=console, show_path=False, markup=True)
    console_handler.setLevel(logging.WARNING)

    root.addHandler(file_handler)
    root.addHandler(console_handler)


def print_banner() -> None:
    model = CONFIG.anthropic_model if CONFIG.llm_backend == "claude" else CONFIG.model

    info = Table.grid(padding=(0, 1))
    info.add_column(style="field", justify="right")
    info.add_column(style="value")
    info.add_row("Backend:", CONFIG.llm_backend)
    info.add_row("Model:", model)
    info.add_row("Vault:", str(CONFIG.vault_dir))
    info.add_row("Web search:", CONFIG.web_search_backend)

    console.print(
        Panel(
            info,
            title="[banner.title]Local AI Assistant[/banner.title]",
            subtitle="[muted]Type 'exit' or 'quit' to leave[/muted]",
            border_style="banner.border",
            padding=(1, 2),
        )
    )


def run_turn_with_preview(agent: Agent, user_message: str) -> str:
    """Run one agent turn, live-rendering streamed tokens as a
    lightweight preview so the terminal shows the model "thinking" in
    real time instead of just a spinner. Tool calls show a short
    status line rather than their raw JSON; the caller is expected to
    render the returned answer properly afterwards (see print_answer)
    -- this preview is discarded once the turn completes.
    """
    preview = Text()
    status_lines: list[str] = []

    def render() -> Panel:
        body = Text()
        for line in status_lines:
            body.append(line + "\n", style="muted")
        body.append(preview)
        return Panel(
            body,
            title="[assistant.title]Assistant[/assistant.title]",
            border_style="assistant.border",
            padding=(0, 1),
        )

    with Live(render(), console=console, refresh_per_second=12, transient=True) as live:

        def on_event(event) -> None:
            nonlocal preview
            if isinstance(event, TokenEvent):
                preview.append(event.text)
            elif isinstance(event, ToolCallEvent):
                args = ", ".join(f"{k}={v!r}" for k, v in event.args.items())
                status_lines.append(f"→ using tool: {event.tool}({args})")
                preview = Text()
            elif isinstance(event, ToolResultEvent):
                pass  # the status line already announced the call; result feeds the next iteration
            live.update(render())

        return agent.run_turn(user_message, on_event=on_event)


def print_answer(answer: str) -> None:
    console.print(
        Panel(
            Markdown(answer),
            title="[assistant.title]Assistant[/assistant.title]",
            border_style="assistant.border",
            padding=(0, 1),
        )
    )


def main() -> None:
    configure_logging()
    logger = logging.getLogger("assistant.main")

    print_banner()
    agent = Agent()

    while True:
        console.print()
        try:
            user_message = console.input("[user.label]You:[/user.label] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[muted]Goodbye.[/muted]")
            break

        if not user_message:
            continue
        if user_message.lower() in {"exit", "quit"}:
            console.print("[muted]Goodbye.[/muted]")
            break

        try:
            answer = run_turn_with_preview(agent, user_message)
        except KeyboardInterrupt:
            # Ctrl+C during a long-running (or stuck) request -- e.g. a
            # slow local model. Cancel just this turn, don't crash the
            # REPL: KeyboardInterrupt is a BaseException, not an
            # Exception, so it isn't caught by the except below.
            console.print("\n[muted]Cancelled.[/muted]")
            continue
        except LLMConnectionError as exc:
            console.print(f"[error]Error:[/error] {exc}")
            continue
        except Exception:  # noqa: BLE001 - never let the CLI crash outright
            logger.exception("Unexpected error handling user turn")
            console.print("[error]Something went wrong on my end[/error] -- check assistant.log for details.")
            continue

        print_answer(answer)


if __name__ == "__main__":
    main()
