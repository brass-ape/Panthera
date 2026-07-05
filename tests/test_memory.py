from __future__ import annotations

import pytest

from memory import VaultError, VaultMemory


@pytest.fixture
def memory(vault_dir):
    return VaultMemory()


class TestBasicFileOps:
    def test_write_then_read(self, memory):
        memory.write("facts/linux.md", "Linux is a kernel.")
        assert memory.read("facts/linux.md") == "Linux is a kernel."

    def test_write_creates_parent_folders(self, memory, vault_dir):
        memory.write("projects/assistant/notes.md", "hello")
        assert (vault_dir / "projects" / "assistant" / "notes.md").exists()

    def test_read_missing_file_raises(self, memory):
        with pytest.raises(VaultError):
            memory.read("facts/does-not-exist.md")

    def test_append_to_new_file_creates_it(self, memory):
        memory.append("journal/today.md", "First entry.")
        assert memory.read("journal/today.md").strip() == "First entry."

    def test_append_adds_newline_separator(self, memory):
        memory.write("facts/a.md", "line one")
        memory.append("facts/a.md", "line two")
        content = memory.read("facts/a.md")
        assert content == "line one\nline two\n"

    def test_remove_deletes_file(self, memory):
        memory.write("facts/temp.md", "temp")
        memory.remove("facts/temp.md")
        with pytest.raises(VaultError):
            memory.read("facts/temp.md")

    def test_remove_missing_file_raises(self, memory):
        with pytest.raises(VaultError):
            memory.remove("facts/never-existed.md")

    def test_create_folder(self, memory, vault_dir):
        memory.create_folder("projects/newthing")
        assert (vault_dir / "projects" / "newthing").is_dir()

    def test_read_multiple_reports_per_file_errors(self, memory):
        memory.write("facts/a.md", "A")
        results = memory.read_multiple(["facts/a.md", "facts/missing.md"])
        assert results["facts/a.md"] == "A"
        assert "error" in results["facts/missing.md"]

    def test_list_files_returns_vault_relative_paths(self, memory):
        memory.write("facts/a.md", "a")
        memory.write("people/bob.md", "b")
        files = memory.list_files()
        assert "facts/a.md" in files
        assert "people/bob.md" in files

    def test_list_files_scoped_to_folder(self, memory):
        memory.write("facts/a.md", "a")
        memory.write("people/bob.md", "b")
        files = memory.list_files("facts")
        assert files == ["facts/a.md"]


class TestKeywordSearch:
    def test_search_finds_content_match(self, memory):
        memory.write("facts/rust.md", "Rust is a systems programming language.")
        hits = memory.search("rust programming")
        assert any(h.path == "facts/rust.md" for h in hits)

    def test_search_no_match_returns_empty(self, memory):
        memory.write("facts/rust.md", "Rust is a systems programming language.")
        assert memory.search("nonexistent gibberish query") == []

    def test_search_ranks_filename_match_higher(self, memory):
        memory.write("facts/python.md", "some unrelated content about snakes")
        memory.write("facts/other.md", "this file just mentions python in passing")
        hits = memory.search("python")
        assert hits[0].path == "facts/python.md"

    def test_retrieve_context_for_returns_path_content_pairs(self, memory):
        memory.write("facts/a.md", "a distinctive keyword here")
        context = memory.retrieve_context_for("distinctive keyword")
        assert context
        path, content = context[0]
        assert path == "facts/a.md"
        assert "distinctive keyword" in content

    def test_retrieve_context_for_no_match_returns_empty(self, memory):
        memory.write("facts/a.md", "totally unrelated content")
        assert memory.retrieve_context_for("nonexistent gibberish query") == []


class TestReadResources:
    def test_returns_empty_when_no_resources(self, memory):
        assert memory.read_resources() == []

    def test_reads_all_files_regardless_of_relevance(self, memory):
        memory.write("resources/brief.md", "project brief content")
        memory.write("resources/style.md", "style guide content")
        results = dict(memory.read_resources())
        assert results["resources/brief.md"] == "project brief content"
        assert results["resources/style.md"] == "style guide content"

    def test_capped_at_max_resource_files(self, memory, config_override):
        import memory as memory_module

        for i in range(5):
            memory.write(f"resources/note{i}.md", f"content {i}")

        config_override(memory_module, max_resource_files=2)
        assert len(memory.read_resources()) == 2
