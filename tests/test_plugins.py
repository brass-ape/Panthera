from __future__ import annotations

import pytest

import plugins as plugins_module
from plugins import PluginError, load_approved_plugins, register_loaded_plugins


VALID_PLUGIN_SOURCE = '''\
TOOL_NAME = "test_echo"
REQUIRED_ARGS = ("text",)
DESCRIPTION = "Echoes text back."


def handle(memory, args):
    return f"echo: {args['text']}"
'''

MISSING_ATTR_PLUGIN_SOURCE = '''\
TOOL_NAME = "test_broken"
REQUIRED_ARGS = ("x",)
# DESCRIPTION missing entirely


def handle(memory, args):
    return "should not load"
'''

BAD_REQUIRED_ARGS_PLUGIN_SOURCE = '''\
TOOL_NAME = "test_bad_args"
REQUIRED_ARGS = ["x", "y"]  # must be a tuple, not a list
DESCRIPTION = "Bad shape."


def handle(memory, args):
    return "should not load"
'''

RAISES_AT_IMPORT_PLUGIN_SOURCE = '''\
raise RuntimeError("boom during import")
'''


@pytest.fixture
def plugins_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(plugins_module, "PLUGINS_DIR", tmp_path)
    monkeypatch.setattr(plugins_module, "_registered_specs", None)
    return tmp_path


class TestLoadApprovedPlugins:
    def test_no_directory_returns_empty(self, plugins_dir, monkeypatch):
        monkeypatch.setattr(plugins_module, "PLUGINS_DIR", plugins_dir / "does-not-exist")
        assert load_approved_plugins() == []

    def test_loads_a_valid_plugin(self, plugins_dir):
        (plugins_dir / "echo.py").write_text(VALID_PLUGIN_SOURCE)
        specs = load_approved_plugins()
        assert len(specs) == 1
        assert specs[0].tool_name == "test_echo"
        assert specs[0].required_args == ("text",)
        assert specs[0].description == "Echoes text back."
        assert specs[0].handle(None, {"text": "hi"}) == "echo: hi"

    def test_skips_plugin_missing_required_attribute(self, plugins_dir):
        (plugins_dir / "broken.py").write_text(MISSING_ATTR_PLUGIN_SOURCE)
        assert load_approved_plugins() == []

    def test_skips_plugin_with_wrong_required_args_type(self, plugins_dir):
        (plugins_dir / "bad_args.py").write_text(BAD_REQUIRED_ARGS_PLUGIN_SOURCE)
        assert load_approved_plugins() == []

    def test_skips_plugin_that_raises_on_import(self, plugins_dir):
        (plugins_dir / "raises.py").write_text(RAISES_AT_IMPORT_PLUGIN_SOURCE)
        assert load_approved_plugins() == []

    def test_ignores_underscore_prefixed_files(self, plugins_dir):
        (plugins_dir / "_helpers.py").write_text("TOOL_NAME = 'nope'\n")
        assert load_approved_plugins() == []

    def test_one_broken_plugin_does_not_block_a_valid_one(self, plugins_dir):
        (plugins_dir / "broken.py").write_text(RAISES_AT_IMPORT_PLUGIN_SOURCE)
        (plugins_dir / "echo.py").write_text(VALID_PLUGIN_SOURCE)
        specs = load_approved_plugins()
        assert len(specs) == 1
        assert specs[0].tool_name == "test_echo"


class TestRegisterLoadedPlugins:
    def test_registers_into_parser_and_tools(self, plugins_dir):
        import parser as parser_module
        import tools as tools_module

        (plugins_dir / "echo.py").write_text(VALID_PLUGIN_SOURCE)
        try:
            register_loaded_plugins()
            assert "test_echo" in parser_module.VALID_TOOLS
            assert parser_module._REQUIRED_ARGS["test_echo"] == ("text",)
            assert "test_echo" in tools_module._HANDLERS
            assert tools_module._HANDLERS["test_echo"](None, {"text": "yo"}) == "echo: yo"
        finally:
            parser_module.VALID_TOOLS.discard("test_echo")
            parser_module._REQUIRED_ARGS.pop("test_echo", None)
            tools_module._HANDLERS.pop("test_echo", None)

    def test_is_idempotent_across_calls(self, plugins_dir):
        import parser as parser_module
        import tools as tools_module

        (plugins_dir / "echo.py").write_text(VALID_PLUGIN_SOURCE)
        try:
            first = register_loaded_plugins()
            second = register_loaded_plugins()
            assert first is second
        finally:
            parser_module.VALID_TOOLS.discard("test_echo")
            parser_module._REQUIRED_ARGS.pop("test_echo", None)
            tools_module._HANDLERS.pop("test_echo", None)

    def test_skips_plugin_colliding_with_a_builtin_tool_name(self, plugins_dir):
        import parser as parser_module

        collide_source = VALID_PLUGIN_SOURCE.replace("test_echo", "read")
        (plugins_dir / "collide.py").write_text(collide_source)
        register_loaded_plugins()
        # "read" is a real built-in handler, must not have been
        # clobbered by the plugin's handle().
        import tools as tools_module

        assert tools_module._HANDLERS["read"].__module__ == "tools"
        assert "read" in parser_module.VALID_TOOLS


class TestDescribeLoadedPlugins:
    def test_empty_list_yields_empty_string(self):
        assert plugins_module.describe_loaded_plugins([]) == ""

    def test_describes_each_plugin(self, plugins_dir):
        (plugins_dir / "echo.py").write_text(VALID_PLUGIN_SOURCE)
        specs = load_approved_plugins()
        description = plugins_module.describe_loaded_plugins(specs)
        assert "test_echo" in description
        assert "Echoes text back." in description
        assert '"tool": "test_echo"' in description
