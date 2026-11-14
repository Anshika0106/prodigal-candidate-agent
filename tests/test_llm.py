"""GeminiChat logic tested without the network (SDK objects faked)."""
from types import SimpleNamespace as NS

import pytest
from google.genai import errors, types

import llm
from llm import GeminiChat, LLMError, ToolCall, _to_schema
from tools import TOOL_SPECS


def make_chat(*outcomes):
    chat = GeminiChat.__new__(GeminiChat)
    chat._types, chat._errors = types, errors
    sleeps = []
    chat._sleep = sleeps.append
    queue = list(outcomes)

    def send_message(message):
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item
    chat._chat = NS(send_message=send_message)
    return chat, sleeps


def response(*parts):
    return NS(candidates=[NS(content=NS(parts=list(parts)), finish_reason="STOP")])


def text_part(t, thought=False):
    return NS(text=t, function_call=None, thought=thought)


def call_part(name, args, id="x1"):
    return NS(text=None, function_call=NS(name=name, args=args, id=id), thought=False)


def test_parse_text_and_function_calls_and_skips_thoughts():
    chat, _ = make_chat(response(text_part("thinking...", thought=True), text_part("Hello "),
                                 text_part("there"), call_part("save_progress", {}),
                                 call_part("record_answers", {"answers": {"cgpa": "8"}}, "z")))
    turn = chat.send_text("hi")
    assert turn.text == "Hello there"
    assert [(c.name, c.args, c.id) for c in turn.calls] == [
        ("save_progress", {}, "x1"), ("record_answers", {"answers": {"cgpa": "8"}}, "z")]


def test_transient_errors_are_retried_then_succeed():
    boom = errors.APIError(503, {"error": {"message": "busy"}})
    chat, sleeps = make_chat(boom, boom, response(text_part("ok")))
    assert chat.send_text("hi").text == "ok" and len(sleeps) == 2


def test_persistent_and_non_retryable_errors_raise_llm_error():
    boom = errors.APIError(503, {"error": {"message": "busy"}})
    chat, sleeps = make_chat(boom, boom, boom, boom)
    with pytest.raises(LLMError, match="503"):
        chat.send_text("hi")
    assert len(sleeps) == len(llm.RETRY_DELAYS)
    bad_key = errors.APIError(400, {"error": {"message": "API key not valid sk-SECRET"}})
    chat, sleeps = make_chat(bad_key)
    with pytest.raises(LLMError) as err:
        chat.send_text("hi")
    assert "SECRET" not in str(err.value) and sleeps == []      # no retry, no leaked detail


def test_network_exception_is_wrapped():
    chat, _ = make_chat(*[ConnectionError("down")] * 4)
    with pytest.raises(LLMError, match="ConnectionError"):
        chat.send_text("hi")


def test_empty_response_raises():
    chat, _ = make_chat(NS(candidates=[]))
    with pytest.raises(LLMError):
        chat.send_text("hi")


def test_tool_results_become_function_response_parts():
    sent = []
    chat, _ = make_chat(response(text_part("done")))
    original = chat._chat.send_message
    chat._chat = NS(send_message=lambda m: (sent.append(m), original(m))[1])
    chat.send_tool_results([(ToolCall("save_progress", {}, "abc"), {"saved_fields": ["email"]})])
    part = sent[0][0]
    assert part.function_response.name == "save_progress" and part.function_response.id == "abc"
    assert part.function_response.response == {"result": {"saved_fields": ["email"]}}


def test_every_tool_spec_converts_to_a_gemini_declaration():
    for spec in TOOL_SPECS:
        params = spec["parameters"]
        declaration = types.FunctionDeclaration(
            name=spec["name"], description=spec["description"],
            parameters=_to_schema(params, types) if params.get("properties") else None)
        assert declaration.name == spec["name"]
    record = next(s for s in TOOL_SPECS if s["name"] == "record_answers")
    schema = _to_schema(record["parameters"], types)
    assert schema.required == ["answers"]
    assert "cgpa" in schema.properties["answers"].properties
    assert schema.properties["answers"].properties["skills"].type == types.Type.ARRAY
    assert "phone" not in schema.properties["answers"].properties         # never offered to the LLM
    assert "full_name" not in schema.properties["answers"].properties
