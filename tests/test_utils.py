from __future__ import annotations

import pytest

from utils import UnsafePathError, resolve_safe_path, truncate, word_tokens


class TestResolveSafePath:
    def test_accepts_simple_relative_path(self, vault_dir):
        target = resolve_safe_path("people/me.md")
        assert target == vault_dir / "people" / "me.md"

    def test_rejects_parent_traversal(self, vault_dir):
        with pytest.raises(UnsafePathError):
            resolve_safe_path("../escape.md")

    def test_rejects_nested_traversal(self, vault_dir):
        with pytest.raises(UnsafePathError):
            resolve_safe_path("people/../../escape.md")

    def test_rejects_absolute_posix_path(self, vault_dir):
        with pytest.raises(UnsafePathError):
            resolve_safe_path("/etc/passwd")

    def test_rejects_absolute_windows_path(self, vault_dir):
        with pytest.raises(UnsafePathError):
            resolve_safe_path("C:\\Windows\\system.ini")

    def test_rejects_empty_path(self, vault_dir):
        with pytest.raises(UnsafePathError):
            resolve_safe_path("")

    def test_rejects_none_path(self, vault_dir):
        with pytest.raises(UnsafePathError):
            resolve_safe_path(None)

    def test_rejects_null_byte(self, vault_dir):
        with pytest.raises(UnsafePathError):
            resolve_safe_path("facts/evil\x00.md")

    def test_rejects_disallowed_characters(self, vault_dir):
        with pytest.raises(UnsafePathError):
            resolve_safe_path("facts/$(rm -rf).md")

    def test_rejects_symlink_escaping_vault(self, vault_dir):
        outside = vault_dir.parent / "outside.md"
        outside.write_text("secret")
        link = vault_dir / "link.md"
        link.symlink_to(outside)
        with pytest.raises(UnsafePathError):
            resolve_safe_path("link.md")

    def test_allows_symlink_inside_vault(self, vault_dir):
        (vault_dir / "facts").mkdir(exist_ok=True)
        real = vault_dir / "facts" / "real.md"
        real.write_text("hello")
        link = vault_dir / "link.md"
        link.symlink_to(real)
        # Should not raise -- the symlink resolves inside the vault.
        resolve_safe_path("link.md")


class TestTruncate:
    def test_returns_text_unchanged_when_under_limit(self):
        assert truncate("short", 100) == "short"

    def test_truncates_and_marks_long_text(self):
        result = truncate("a" * 200, 10)
        assert result.startswith("a" * 10)
        assert result.endswith("[truncated]")
        assert len(result) < 200


class TestWordTokens:
    def test_lowercases_and_splits_on_non_alnum(self):
        assert word_tokens("Hello, World! 123") == {"hello", "world", "123"}

    def test_empty_string_yields_no_tokens(self):
        assert word_tokens("") == set()
