"""Gemini chat wrapper using the official Google Gen AI SDK (genuine function calling).

The agent talks to the small `LLM` interface (send_text / send_tool_results -> LLMTurn),
so tests can substitute a scripted fake. Automatic function calling is DISABLED: the
agent executes tools itself so every call passes through Python validation.
"""
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

RETRYABLE_CODES = {429, 500, 503, 504}
RETRY_DELAYS = (3, 8, 20)          # seconds between attempts on transient errors
REQUEST_TIMEOUT_MS = 45_000


class LLMError(Exception):
    """The model could not be reached or returned nothing usable. Message is safe to print."""


@dataclass
class ToolCall:
    name: str
    args: dict
    id: str | None = None


@dataclass
class LLMTurn:
    text: str = ""
    calls: list[ToolCall] = field(default_factory=list)


class LLM(Protocol):
    def send_text(self, text: str) -> LLMTurn: ...
    def send_tool_results(self, results: list[tuple[ToolCall, dict]]) -> LLMTurn: ...


def _to_schema(spec: dict, types) -> Any:
    """Convert a provider-neutral JSON-schema dict into a google.genai Schema."""
    kwargs: dict[str, Any] = {"type": getattr(types.Type, spec["type"].upper())}
    if "description" in spec:
        kwargs["description"] = spec["description"]
    if "enum" in spec:
        kwargs["enum"] = list(spec["enum"])
    if "properties" in spec and spec["properties"]:
        kwargs["properties"] = {k: _to_schema(v, types) for k, v in spec["properties"].items()}
    if "items" in spec:
        kwargs["items"] = _to_schema(spec["items"], types)
    if "required" in spec:
        kwargs["required"] = list(spec["required"])
    return types.Schema(**kwargs)


class GeminiChat:
    def __init__(self, api_key: str, model: str, system_instruction: str, tool_specs: list[dict],
                 temperature: float = 0.3, sleep: Callable[[float], None] = time.sleep):
        from google import genai
        from google.genai import errors, types

        self._types, self._errors, self._sleep = types, errors, sleep
        self.model = model
        declarations = []
        for spec in tool_specs:
            params = spec.get("parameters") or {"type": "object", "properties": {}}
            declarations.append(types.FunctionDeclaration(
                name=spec["name"], description=spec["description"],
                # Empty-parameter tools must omit the schema entirely.
                parameters=_to_schema(params, types) if params.get("properties") else None,
            ))
        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=REQUEST_TIMEOUT_MS, retry_options=types.HttpRetryOptions(attempts=1)),
        )
        self._chat = client.chats.create(
            model=model,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=[types.Tool(function_declarations=declarations)],
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                temperature=temperature,
                thinking_config=types.ThinkingConfig(thinking_level="LOW"),
            ),
        )

    def send_text(self, text: str) -> LLMTurn:
        return self._send(text)

    def send_tool_results(self, results: list[tuple[ToolCall, dict]]) -> LLMTurn:
        t = self._types
        parts = [
            t.Part(function_response=t.FunctionResponse(name=call.name, id=call.id, response={"result": out}))
            for call, out in results
        ]
        return self._send(parts)

    def _send(self, message) -> LLMTurn:
        last_code = None
        for attempt in range(len(RETRY_DELAYS) + 1):
            try:
                response = self._chat.send_message(message)
                break
            except self._errors.APIError as exc:
                last_code = getattr(exc, "code", None)
                if last_code in RETRYABLE_CODES and attempt < len(RETRY_DELAYS):
                    self._sleep(RETRY_DELAYS[attempt])
                    continue
                raise LLMError(f"Gemini API error (HTTP {last_code})") from None
            except Exception as exc:  # network failures, timeouts
                if attempt < len(RETRY_DELAYS):
                    self._sleep(RETRY_DELAYS[attempt])
                    continue
                raise LLMError(f"Gemini request failed ({type(exc).__name__})") from None
        return self._parse(response)

    @staticmethod
    def _parse(response) -> LLMTurn:
        candidates = getattr(response, "candidates", None) or []
        content = candidates[0].content if candidates else None
        if content is None or not content.parts:
            reason = getattr(candidates[0], "finish_reason", None) if candidates else "no candidates"
            raise LLMError(f"Gemini returned no content ({reason})")
        text, calls = [], []
        for part in content.parts:
            if getattr(part, "thought", False):
                continue
            if part.function_call:
                calls.append(ToolCall(part.function_call.name, dict(part.function_call.args or {}),
                                      getattr(part.function_call, "id", None)))
            elif part.text:
                text.append(part.text)
        return LLMTurn("".join(text).strip(), calls)
