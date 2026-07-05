from __future__ import annotations

import json

import config as config_module
from config import CONFIG, Config


def _reset_config_cache(monkeypatch, tmp_path):
    """Point config.json resolution at an isolated tmp_path and clear
    the module's cache so each test starts with no overrides on disk.
    """
    monkeypatch.setenv("ASSISTANT_CONFIG_FILE", str(tmp_path / "config.json"))
    monkeypatch.setattr(config_module, "_config_file_cache", None)


class TestSettingHelpers:
    def test_env_var_used_when_no_config_file(self, monkeypatch, tmp_path):
        _reset_config_cache(monkeypatch, tmp_path)
        monkeypatch.setenv("ASSISTANT_MODEL", "llama3.1:8b")
        assert config_module._setting_str("model", "ASSISTANT_MODEL", "default-model") == "llama3.1:8b"

    def test_default_used_when_neither_set(self, monkeypatch, tmp_path):
        _reset_config_cache(monkeypatch, tmp_path)
        monkeypatch.delenv("ASSISTANT_MODEL", raising=False)
        assert config_module._setting_str("model", "ASSISTANT_MODEL", "default-model") == "default-model"

    def test_config_file_overrides_env_var(self, monkeypatch, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"model": "from-config-json"}))
        _reset_config_cache(monkeypatch, tmp_path)
        monkeypatch.setenv("ASSISTANT_MODEL", "from-env-var")
        assert config_module._setting_str("model", "ASSISTANT_MODEL", "default-model") == "from-config-json"

    def test_malformed_config_file_falls_back_to_env(self, monkeypatch, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text("{not valid json")
        _reset_config_cache(monkeypatch, tmp_path)
        monkeypatch.setenv("ASSISTANT_MODEL", "from-env-var")
        assert config_module._setting_str("model", "ASSISTANT_MODEL", "default-model") == "from-env-var"

    def test_setting_int_converts_config_file_string(self, monkeypatch, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"max_tool_iterations": "7"}))
        _reset_config_cache(monkeypatch, tmp_path)
        assert config_module._setting_int("max_tool_iterations", "ASSISTANT_MAX_TOOL_ITERATIONS", 16) == 7

    def test_setting_float_reads_config_file(self, monkeypatch, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"temperature": 0.9}))
        _reset_config_cache(monkeypatch, tmp_path)
        assert config_module._setting_float("temperature", "ASSISTANT_TEMPERATURE", 0.4) == 0.9


class TestSaveAndReload:
    def test_save_overrides_writes_file_and_updates_live_config(self, monkeypatch, tmp_path):
        _reset_config_cache(monkeypatch, tmp_path)
        original_model = CONFIG.model
        try:
            config_module.save_overrides({"model": "reloaded-model"})
            assert CONFIG.model == "reloaded-model"

            written = json.loads((tmp_path / "config.json").read_text())
            assert written["model"] == "reloaded-model"
        finally:
            config_module.save_overrides({"model": original_model})

    def test_save_overrides_merges_with_existing_file(self, monkeypatch, tmp_path):
        _reset_config_cache(monkeypatch, tmp_path)
        try:
            config_module.save_overrides({"model": "model-a"})
            config_module.save_overrides({"temperature": 0.1})

            written = json.loads((tmp_path / "config.json").read_text())
            assert written["model"] == "model-a"
            assert written["temperature"] == 0.1
        finally:
            config_module.save_overrides({"model": Config().model, "temperature": Config().temperature})

    def test_current_overrides_reflects_disk_contents(self, monkeypatch, tmp_path):
        _reset_config_cache(monkeypatch, tmp_path)
        original_log_level = CONFIG.log_level
        try:
            config_module.save_overrides({"log_level": "DEBUG"})
            assert config_module.current_overrides()["log_level"] == "DEBUG"
        finally:
            config_module.save_overrides({"log_level": original_log_level})


class TestEditableFields:
    def test_project_root_is_not_editable(self):
        assert "project_root" not in config_module.EDITABLE_FIELDS

    def test_editable_fields_are_real_config_attributes(self):
        for name in config_module.EDITABLE_FIELDS:
            assert hasattr(CONFIG, name)

    def test_no_secret_fields_are_editable(self):
        # "_tokens" (plural, e.g. anthropic_max_tokens) means a count
        # limit, not a credential -- only flag the singular/credential
        # forms.
        for name in config_module.EDITABLE_FIELDS:
            assert not name.endswith(("_key", "_secret", "_token"))
            assert "api_key" not in name and "secret" not in name
