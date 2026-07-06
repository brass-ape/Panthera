from __future__ import annotations

from parser import ToolCallValidationError, parse_llm_response, validate_tool_call


class TestParseLlmResponse:
    def test_parses_bare_json_tool_call(self):
        parsed = parse_llm_response('{"tool": "read", "file": "people/me.md"}')
        assert parsed.is_tool_call
        assert parsed.tool_call.tool == "read"
        assert parsed.tool_call.args == {"file": "people/me.md"}

    def test_tolerates_raw_newlines_inside_json_string_values(self):
        # Small local models very commonly emit a multi-line "content"
        # field as a literal newline instead of an escaped "\n" -- that
        # is invalid strict JSON but unambiguous in intent, and should
        # still parse as a real tool call rather than falling back to
        # dumping the raw (garbled-looking) JSON as the final answer.
        text = '{"tool": "write", "file": "facts/sade.md", "content": "# Sade\nline two\nline three"}'
        parsed = parse_llm_response(text)
        assert parsed.is_tool_call
        assert parsed.tool_call.tool == "write"
        assert parsed.tool_call.args["content"] == "# Sade\nline two\nline three"

    def test_parses_json_wrapped_in_markdown_fence(self):
        text = '```json\n{"tool": "search", "query": "rust"}\n```'
        parsed = parse_llm_response(text)
        assert parsed.is_tool_call
        assert parsed.tool_call.tool == "search"
        assert parsed.tool_call.args == {"query": "rust"}

    def test_parses_bare_fence_without_language_tag(self):
        text = '```\n{"tool": "list_files"}\n```'
        parsed = parse_llm_response(text)
        assert parsed.is_tool_call
        assert parsed.tool_call.tool == "list_files"

    def test_plain_prose_is_final_text(self):
        parsed = parse_llm_response("Hello! I don't need any tools right now.")
        assert not parsed.is_tool_call
        assert parsed.final_text == "Hello! I don't need any tools right now."

    def test_prose_merely_mentioning_braces_is_not_parsed_as_json(self):
        text = "You can set config like {this} in your file, but I have no tool to run."
        parsed = parse_llm_response(text)
        assert not parsed.is_tool_call
        assert parsed.final_text == text

    def test_malformed_json_falls_back_to_final_text(self):
        text = '{"tool": "read", "file": '
        parsed = parse_llm_response(text)
        assert not parsed.is_tool_call
        assert parsed.final_text == text.strip()

    def test_unknown_tool_falls_back_to_final_text(self):
        parsed = parse_llm_response('{"tool": "delete_everything"}')
        assert not parsed.is_tool_call

    def test_missing_required_arg_falls_back_to_final_text(self):
        parsed = parse_llm_response('{"tool": "read"}')
        assert not parsed.is_tool_call

    def test_json_array_is_not_a_tool_call(self):
        parsed = parse_llm_response('["not", "an", "object"]')
        assert not parsed.is_tool_call

    def test_web_search_tool_call(self):
        parsed = parse_llm_response('{"tool": "web_search", "query": "python news"}')
        assert parsed.is_tool_call
        assert parsed.tool_call.tool == "web_search"
        assert parsed.tool_call.args == {"query": "python news"}

    def test_web_fetch_tool_call(self):
        parsed = parse_llm_response('{"tool": "web_fetch", "url": "https://example.com"}')
        assert parsed.is_tool_call
        assert parsed.tool_call.tool == "web_fetch"
        assert parsed.tool_call.args == {"url": "https://example.com"}

    def test_web_fetch_missing_url_falls_back(self):
        parsed = parse_llm_response('{"tool": "web_fetch"}')
        assert not parsed.is_tool_call

    def test_propose_plugin_tool_call(self):
        parsed = parse_llm_response(
            '{"tool": "propose_plugin", "name": "roll_dice", '
            '"description": "Rolls a die", "code": "TOOL_NAME = 1"}'
        )
        assert parsed.is_tool_call
        assert parsed.tool_call.tool == "propose_plugin"
        assert parsed.tool_call.args["name"] == "roll_dice"

    def test_propose_plugin_rejects_bad_name(self):
        parsed = parse_llm_response(
            '{"tool": "propose_plugin", "name": "../escape", '
            '"description": "x", "code": "y"}'
        )
        assert not parsed.is_tool_call

    def test_aliased_tool_name_resolves_to_real_tool(self):
        # Models trained on other agent frameworks sometimes reach for
        # a plausible-sounding synonym instead of this app's real tool
        # names -- see parser._TOOL_ALIASES.
        parsed = parse_llm_response(
            '{"tool": "create_file", "file": "facts/a.md", "content": "hi"}'
        )
        assert parsed.is_tool_call
        assert parsed.tool_call.tool == "write"
        assert parsed.tool_call.args == {"file": "facts/a.md", "content": "hi"}

    def test_aliased_tool_still_requires_real_tools_args(self):
        # "delete_file" aliases to "remove", which requires "file" --
        # aliasing doesn't bypass the real tool's argument schema.
        parsed = parse_llm_response('{"tool": "delete_file"}')
        assert not parsed.is_tool_call

    def test_unrecognized_tool_name_is_not_aliased(self):
        parsed = parse_llm_response('{"tool": "delete_everything_forever"}')
        assert not parsed.is_tool_call

    def test_propose_plugin_rejects_uppercase_name(self):
        parsed = parse_llm_response(
            '{"tool": "propose_plugin", "name": "RollDice", '
            '"description": "x", "code": "y"}'
        )
        assert not parsed.is_tool_call


class TestValidateToolCall:
    def test_missing_tool_field_raises(self):
        try:
            validate_tool_call({})
            assert False, "expected ToolCallValidationError"
        except ToolCallValidationError:
            pass

    def test_read_multiple_requires_list_of_strings(self):
        with_bad_files = {"tool": "read_multiple", "files": "not-a-list"}
        try:
            validate_tool_call(with_bad_files)
            assert False, "expected ToolCallValidationError"
        except ToolCallValidationError:
            pass

    def test_read_multiple_accepts_list_of_strings(self):
        call = validate_tool_call({"tool": "read_multiple", "files": ["a.md", "b.md"]})
        assert call.tool == "read_multiple"
        assert call.args == {"files": ["a.md", "b.md"]}

    def test_none_tool_needs_no_args(self):
        call = validate_tool_call({"tool": "none"})
        assert call.tool == "none"
        assert call.args == {}
