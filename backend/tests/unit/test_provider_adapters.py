"""Adapter tests against a mocked HTTP transport (no Ollama/cloud needed)."""

import json

import httpx
import pytest

from app.domain.ai.capabilities import StructuredOutputMode
from app.domain.ai.exceptions import (
    ProviderError,
    ProviderRequestError,
    ProviderTimeout,
    ProviderUnavailable,
    RateLimitExceeded,
    StructuredOutputError,
)
from app.domain.ai.generation import ChatMessage
from app.domain.cards.schemas import GeneratedCardBatch
from app.infrastructure.ai.providers.ollama_embedding import OllamaEmbeddingProvider
from app.infrastructure.ai.providers.ollama_generation import OllamaGenerationProvider
from app.infrastructure.ai.providers.openai_compatible import OpenAICompatibleGenerationProvider
from tests.helpers import tr_batch

MESSAGES = [ChatMessage.system("s"), ChatMessage.user("u")]


def client_for(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def ollama_reply(content, **extra):
    return httpx.Response(
        200,
        json={
            "message": {"role": "assistant", "content": content},
            "done": True,
            "done_reason": extra.get("done_reason", "stop"),
            "prompt_eval_count": 120,
            "eval_count": 80,
        },
    )


async def test_ollama_structured_request_and_normalized_result():
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return ollama_reply(tr_batch("Kahve"))

    provider = OllamaGenerationProvider(client=client_for(handler), base_url="http://ollama:11434/")
    result = await provider.generate_structured(
        model="some-model",
        messages=MESSAGES,
        schema=GeneratedCardBatch,
        temperature=0.4,
        max_tokens=500,
        options={"think": False, "keep_alive": "5m", "num_ctx": 4096},
    )
    body = seen["body"]
    assert seen["url"] == "http://ollama:11434/api/chat"
    assert body["format"]["type"] == "object"  # JSON schema passed to `format`
    assert body["think"] is False and body["keep_alive"] == "5m"
    assert body["options"] == {"num_ctx": 4096, "temperature": 0.4, "num_predict": 500}
    assert body["stream"] is False
    assert body["messages"][0] == {"role": "system", "content": "s"}
    assert result.parsed.cards[0].target == "Kahve"
    assert (result.input_tokens, result.output_tokens) == (120, 80)
    assert (result.provider, result.model, result.raw_finish_reason) == (
        "ollama",
        "some-model",
        "stop",
    )


async def test_ollama_json_mode_and_prompt_only_formats():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return ollama_reply(tr_batch("Kahve"))

    provider = OllamaGenerationProvider(client=client_for(handler), base_url="http://o")
    for mode in (StructuredOutputMode.JSON_MODE, StructuredOutputMode.PROMPT_ONLY):
        await provider.generate_structured(
            model="m", messages=MESSAGES, schema=GeneratedCardBatch, temperature=0, mode=mode
        )
    assert bodies[0]["format"] == "json"
    assert "format" not in bodies[1]


async def test_ollama_truncated_answer_is_a_structured_output_error():
    provider = OllamaGenerationProvider(
        client=client_for(lambda r: ollama_reply('{"cards": [', done_reason="length")),
        base_url="http://o",
    )
    with pytest.raises(StructuredOutputError) as info:
        await provider.generate_structured(
            model="m", messages=MESSAGES, schema=GeneratedCardBatch, temperature=0
        )
    assert info.value.result.output_tokens == 80


@pytest.mark.parametrize(
    ("response", "error"),
    [
        (httpx.Response(500, text="internal secret stack trace"), ProviderUnavailable),
        (httpx.Response(503), ProviderUnavailable),
        (httpx.Response(429, headers={"retry-after": "7"}), RateLimitExceeded),
        (httpx.Response(404, json={"error": "model 'x' not found"}), ProviderRequestError),
        (httpx.Response(504), ProviderTimeout),
        (httpx.Response(200, text="<html>"), ProviderError),
    ],
)
async def test_ollama_http_errors_are_normalized(response, error):
    provider = OllamaGenerationProvider(client=client_for(lambda r: response), base_url="http://o")
    with pytest.raises(error) as info:
        await provider.generate_text(model="m", messages=MESSAGES, temperature=0)
    assert "secret" not in str(info.value)
    if error is RateLimitExceeded:
        assert info.value.retry_after == 7


async def test_ollama_transport_errors_are_normalized():
    def refuse(request):
        raise httpx.ConnectError("connection refused")

    def slow(request):
        raise httpx.ReadTimeout("timeout")

    for handler, error in ((refuse, ProviderUnavailable), (slow, ProviderTimeout)):
        provider = OllamaGenerationProvider(client=client_for(handler), base_url="http://o")
        with pytest.raises(error):
            await provider.generate_text(model="m", messages=MESSAGES, temperature=0)
        assert await provider.health_check() is False


async def test_ollama_embedding_adapter():
    def handler(request):
        body = json.loads(request.content)
        assert body["input"] == ["a", "b"] and body["model"] == "embed"
        return httpx.Response(
            200, json={"embeddings": [[0.1, 0.2], [0.3, 0.4]], "prompt_eval_count": 4}
        )

    provider = OllamaEmbeddingProvider(client=client_for(handler), base_url="http://o")
    result = await provider.embed(model="embed", texts=["a", "b"])
    assert result.dimension == 2 and result.vectors[1] == [0.3, 0.4] and result.input_tokens == 4


async def test_ollama_embedding_rejects_inconsistent_dimensions():
    provider = OllamaEmbeddingProvider(
        client=client_for(lambda r: httpx.Response(200, json={"embeddings": [[1.0], [1.0, 2.0]]})),
        base_url="http://o",
    )
    with pytest.raises(ProviderError):
        await provider.embed(model="e", texts=["a", "b"])


async def test_openai_compatible_adapter_payload_and_usage():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": tr_batch("Robot")}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 30},
            },
        )

    provider = OpenAICompatibleGenerationProvider(
        name="cloud_x",
        client=client_for(handler),
        base_url="https://api.example/v1",
        api_key="sk-test",
    )
    result = await provider.generate_structured(
        model="model-x",
        messages=MESSAGES,
        schema=GeneratedCardBatch,
        temperature=0.2,
        max_tokens=300,
    )
    assert seen["auth"] == "Bearer sk-test"
    assert seen["body"]["response_format"]["type"] == "json_schema"
    assert seen["body"]["response_format"]["json_schema"]["name"] == "GeneratedCardBatch"
    assert seen["body"]["max_tokens"] == 300
    assert result.parsed.cards[0].target == "Robot"
    assert (result.input_tokens, result.output_tokens, result.provider) == (50, 30, "cloud_x")


async def test_ollama_crashed_runner_response_is_an_outage():
    # Observed live: after a GPU out-of-memory error Ollama answers HTTP 200 with
    # {"model": "", "done": false} and no message.
    for body in (
        {"model": "", "created_at": "0001-01-01T00:00:00Z", "done": False},
        {"error": "Compute error."},
    ):
        provider = OllamaGenerationProvider(
            client=client_for(lambda r, b=body: httpx.Response(200, json=b)), base_url="http://o"
        )
        with pytest.raises(ProviderUnavailable):
            await provider.generate_text(model="m", messages=MESSAGES, temperature=0)
