"""
sysinfo.py
==========

Read-only environment facts injected into the system prompt every
turn: the current local date/time and basic hardware specs. This is
the only thing the assistant ever knows without a tool call or vault
retrieval -- it's not user data, so it's safe (and useful, e.g. for
"what's today's date" or "will this run on my machine") to include
unconditionally rather than gating it behind a tool.
"""

from __future__ import annotations

import os
import platform
from datetime import datetime


def _total_ram_gib() -> str:
    """Best-effort total RAM. Linux-only (reads /proc/meminfo) --
    returns "unknown" elsewhere rather than adding a psutil dependency
    for a single number.
    """
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kib = int(line.split()[1])
                    return f"{kib / (1024 * 1024):.1f} GiB"
    except (OSError, ValueError, IndexError):
        pass
    return "unknown"


def current_datetime_line() -> str:
    """The local date/time, e.g. 'Sunday, 2026-07-05 23:42 (UTC, UTC+0100)'."""
    now = datetime.now().astimezone()
    return now.strftime("%A, %Y-%m-%d %H:%M (%Z, UTC%z)")


def system_specs_line() -> str:
    """e.g. 'Linux 6.8.0-generic, x86_64, 8 CPU cores, 15.6 GiB RAM'."""
    return (
        f"{platform.system()} {platform.release()}, "
        f"{platform.machine()}, {os.cpu_count() or '?'} CPU cores, "
        f"{_total_ram_gib()} RAM"
    )


def context_block() -> str:
    """A short block for the system prompt, regenerated fresh every
    turn so the date/time is always current.
    """
    return f"Current local date/time: {current_datetime_line()}\nSystem: {system_specs_line()}"
