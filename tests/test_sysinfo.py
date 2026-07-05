from __future__ import annotations

import sysinfo


def test_current_datetime_line_is_nonempty_string():
    line = sysinfo.current_datetime_line()
    assert isinstance(line, str)
    assert line.strip()


def test_system_specs_line_includes_cpu_count():
    line = sysinfo.system_specs_line()
    assert "CPU cores" in line


def test_context_block_combines_both_lines():
    block = sysinfo.context_block()
    assert "Current local date/time:" in block
    assert "System:" in block
