# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Sample collection over the OpenAI-compatible completions endpoint.

Uses the MAX ``return_token_ids`` request extension, so the comparison runs
on token ids and never on detokenized text. Requests carry no sampling fields
beyond the token budget and the seed: the two servers must resolve
temperature, top-p and top-k identically for the comparison to mean anything,
and sending neither of them any of those fields is the surest way to get
that.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Mapping, Sequence

import httpx
from typing_extensions import TypeIs

from .samples import Sample

logger = logging.getLogger(__name__)

_MAX_ATTEMPTS = 1
_INITIAL_BACKOFF_S = 0.5
_MAX_BACKOFF_S = 8.0
_CONNECT_TIMEOUT_S = 10.0
_READ_TIMEOUT_S = 120.0
_TOO_MANY_REQUESTS = 429

COMPLETIONS_ENDPOINT = "/v1/completions"
CHAT_ENDPOINT = "/v1/chat/completions"
ENDPOINTS = (CHAT_ENDPOINT, COMPLETIONS_ENDPOINT)

ChatMessages = Sequence[Mapping[str, str]]
"""A chat conversation as ``role``/``content`` mappings."""


def make_client() -> httpx.AsyncClient:
    """Builds an HTTP client suited to long-running completion requests.

    The caller owns the returned client and should use one for the whole run,
    so every request shares a single connection pool.

    Returns:
        A client with a short connect timeout and HTTP/2 disabled.
    """
    return httpx.AsyncClient(
        timeout=httpx.Timeout(_READ_TIMEOUT_S, connect=_CONNECT_TIMEOUT_S),
        http2=False,
    )


async def discover_model_name(client: httpx.AsyncClient, base_url: str) -> str:
    """Returns the name of the model a server is serving.

    Args:
        client: The HTTP client to use.
        base_url: The server root, not the ``/v1`` prefix.

    Returns:
        The id of the first model the server lists.

    Raises:
        RuntimeError: If the server lists no models or answers oddly.
    """
    response = await client.get(f"{_root(base_url)}/v1/models")
    response.raise_for_status()
    payload = _as_object_mapping(response.json(), "response")
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        raise RuntimeError(
            f"{base_url}/v1/models listed no models; is the server still"
            " loading?"
        )
    model = _as_object_mapping(data[0], "data[0]")
    model_id = model.get("id")
    if not isinstance(model_id, str):
        raise RuntimeError(
            f"{base_url}/v1/models returned a model without a string id:"
            f" {model!r}"
        )
    return model_id


_MASK_64 = (1 << 64) - 1
_MASK_63 = (1 << 63) - 1


def request_seed(base: int, index: int) -> int:
    """Derives the seed sent with one request from a server's base.

    Mixes the base and the request index through SplitMix64's finalizer, so
    two servers given adjacent bases draw unrelated seed streams rather than
    streams offset by one. The result is in ``[1, 2**63 - 1]``: the server
    treats ``0`` as unset and would replace it with a random seed.
    """
    z = ((base << 32) ^ index) & _MASK_64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK_64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK_64
    z ^= z >> 31
    return (z & _MASK_63) or 1


RESERVED_REQUEST_KEYS = frozenset(
    {
        "model",
        "prompt",
        "messages",
        "max_tokens",
        "seed",
        "return_token_ids",
        "stream",
    }
)
"""The request body fields the tool sets itself, so callers may not."""


async def collect_samples(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    model: str,
    prompt: Sequence[int] | str | ChatMessages,
    endpoint: str,
    server: str,
    prompt_index: int,
    num_samples: int,
    max_tokens: int,
    seed_base: int,
    concurrency: int,
    request_params: Mapping[str, object] | None = None,
    on_sample: Callable[[], None] | None = None,
) -> list[Sample]:
    """Draws ``num_samples`` independently seeded completions of one prompt.

    A request that gives up fails the whole call rather than yielding a
    partial set.

    Args:
        client: The HTTP client to use, shared across the whole run.
        base_url: The server root, not the ``/v1`` prefix.
        model: The served model name.
        prompt: Token ids, text, or a chat conversation. Ids always go to
            the completions endpoint, being a finished prefix already.
        endpoint: Which endpoint text and conversations are sent to, one of
            :data:`ENDPOINTS`.
        server: Which server this is, ``"baseline"`` or ``"test"``.
        prompt_index: Index of this prompt within the run's prompt set.
        num_samples: How many completions to draw.
        max_tokens: The output token budget per completion.
        seed_base: This server's seed base; see :func:`request_seed`.
        concurrency: How many requests may be in flight at once.
        request_params: Extra fields for every request body, such as
            sampling settings; without them the server applies the model's
            defaults. Keys in :data:`RESERVED_REQUEST_KEYS` are overridden.
        on_sample: Called once per completed request, for progress display.

    Returns:
        The samples, sorted by request index.

    Raises:
        RuntimeError: This includes when a request gives up, when the server
            rejects one, and when a response is unusable.
        ValueError: This includes when a conversation is given for the
            completions endpoint and when ``endpoint`` is unknown.
    """
    assert seed_base >= 0
    if endpoint not in ENDPOINTS:
        raise ValueError(f"unknown endpoint {endpoint!r}")
    sent_prompt_ids = tuple(prompt) if _is_token_ids(prompt) else None
    path, wire_input = _wire_input(prompt, endpoint)
    url = f"{_root(base_url)}{path}"

    semaphore = asyncio.Semaphore(concurrency)
    started_at = time.monotonic()

    async def draw(request_index: int) -> Sample:
        seed = request_seed(seed_base, request_index)
        body: dict[str, object] = {
            **(request_params or {}),
            "model": model,
            **wire_input,
            "max_tokens": max_tokens,
            "seed": seed,
            "return_token_ids": True,
            "stream": False,
        }
        async with semaphore:
            payload, completed_at = await _post_with_retries(client, url, body)
        sample = _sample_from_payload(
            payload,
            server=server,
            prompt_index=prompt_index,
            request_index=request_index,
            seed=seed,
            sent_prompt_ids=sent_prompt_ids,
            completed_at=completed_at,
        )
        if on_sample is not None:
            on_sample()
        return sample

    samples: list[Sample] = await asyncio.gather(
        *(draw(i) for i in range(num_samples))
    )
    elapsed_s = time.monotonic() - started_at
    logger.info(
        "%s prompt %d: %d samples in %.1fs (%.1f req/s)",
        server,
        prompt_index,
        num_samples,
        elapsed_s,
        num_samples / elapsed_s if elapsed_s > 0 else float("inf"),
    )
    return sorted(samples, key=lambda sample: sample.request_index)


def _is_token_ids(
    prompt: Sequence[int] | str | ChatMessages,
) -> TypeIs[Sequence[int]]:
    """Returns whether a prompt is a token id sequence."""
    return (
        not isinstance(prompt, str)
        and len(prompt) > 0
        and isinstance(prompt[0], int)
    )


def _wire_input(
    prompt: Sequence[int] | str | ChatMessages, endpoint: str
) -> tuple[str, dict[str, object]]:
    """Picks the endpoint path and the request field that carries the prompt.

    Ids are a finished prefix, so they take the completions endpoint whatever
    was asked for. Text becomes one user turn on the chat endpoint.
    """
    if _is_token_ids(prompt):
        return COMPLETIONS_ENDPOINT, {"prompt": list(prompt)}
    if isinstance(prompt, str):
        if endpoint == CHAT_ENDPOINT:
            return endpoint, {"messages": [{"role": "user", "content": prompt}]}
        return endpoint, {"prompt": prompt}
    if endpoint != CHAT_ENDPOINT:
        raise ValueError(
            "a chat conversation can only be sent to the chat endpoint"
        )
    return endpoint, {"messages": [dict(message) for message in prompt]}


async def _post_with_retries(
    client: httpx.AsyncClient, url: str, body: Mapping[str, object]
) -> tuple[Mapping[str, object], float]:
    """Posts one completion request, retrying transport errors, 429 and 5xx.

    Any other 4xx is fatal: a retry cannot fix a malformed request or a
    server that lacks the ``return_token_ids`` extension.
    """
    backoff_s = _INITIAL_BACKOFF_S
    last_failure = ""
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            response = await client.post(url, json=body)
        except httpx.TransportError as exc:
            last_failure = f"{type(exc).__name__}: {exc}"
        else:
            received_at = time.monotonic()
            if response.status_code < 400:
                return (
                    _as_object_mapping(response.json(), "response"),
                    received_at,
                )
            detail = f"HTTP {response.status_code}: {response.text[:2000]}"
            if (
                response.status_code < 500
                and response.status_code != _TOO_MANY_REQUESTS
            ):
                raise RuntimeError(f"{url} rejected the request; {detail}")
            last_failure = detail

        if attempt < _MAX_ATTEMPTS:
            logger.warning(
                "%s attempt %d/%d failed (%s); retrying in %.1fs",
                url,
                attempt,
                _MAX_ATTEMPTS,
                last_failure,
                backoff_s,
            )
            await asyncio.sleep(backoff_s)
            backoff_s = min(backoff_s * 2, _MAX_BACKOFF_S)

    raise RuntimeError(
        f"{url} failed after {_MAX_ATTEMPTS} attempts; last failure was"
        f" {last_failure}"
    )


def _sample_from_payload(
    payload: Mapping[str, object],
    *,
    server: str,
    prompt_index: int,
    request_index: int,
    seed: int,
    sent_prompt_ids: tuple[int, ...] | None,
    completed_at: float,
) -> Sample:
    """Builds a sample from a completion response body.

    ``sent_prompt_ids`` is ``None`` when the prompt went out as text, which is
    the one case where the returned prompt ids cannot be checked against it.
    """
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(
            f"completion response carried no choices: {payload!r}"
        )
    choice = _as_object_mapping(choices[0], "choices[0]")

    token_ids = _optional_int_tuple(choice.get("token_ids"), "token_ids")
    prompt_token_ids = _optional_int_tuple(
        choice.get("prompt_token_ids"), "prompt_token_ids"
    )
    if token_ids is None or prompt_token_ids is None:
        missing = [
            name
            for name, value in (
                ("token_ids", token_ids),
                ("prompt_token_ids", prompt_token_ids),
            )
            if value is None
        ]
        raise RuntimeError(
            "the server did not honor return_token_ids: choices[0] is missing"
            f" {', '.join(missing)}. SpecDecCheck needs a server build that"
            " returns token ids."
        )

    finish_reason = choice.get("finish_reason")
    if not isinstance(finish_reason, str):
        raise RuntimeError(
            f"choices[0].finish_reason must be a string, got {finish_reason!r}"
        )

    if sent_prompt_ids is not None and prompt_token_ids != sent_prompt_ids:
        raise RuntimeError(
            "the server altered the prompt: sent"
            f" {len(sent_prompt_ids)} ids, got back"
            f" {len(prompt_token_ids)} that do not match. Comparing"
            " the servers requires both to start from identical prompt ids."
        )

    return Sample(
        server=server,
        prompt_index=prompt_index,
        request_index=request_index,
        seed=seed,
        prompt_token_ids=prompt_token_ids,
        token_ids=token_ids,
        finish_reason=finish_reason,
        completion_tokens=_usage_completion_tokens(payload),
        completed_at=completed_at,
        text=_choice_text(choice),
    )


def _choice_text(choice: Mapping[str, object]) -> str:
    """Returns a choice's decoded text, reasoning before content for chat."""
    text = choice.get("text")
    if isinstance(text, str):
        return text
    message = choice.get("message")
    if not isinstance(message, Mapping):
        return ""
    parts = []
    for key in ("reasoning", "reasoning_content", "content"):
        value = message.get(key)
        if isinstance(value, str) and value:
            parts.append(value)
    return "".join(parts)


def _usage_completion_tokens(payload: Mapping[str, object]) -> int | None:
    """Returns ``usage.completion_tokens``, or ``None`` if unreported."""
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None
    completion_tokens = usage.get("completion_tokens")
    if isinstance(completion_tokens, bool) or not isinstance(
        completion_tokens, int
    ):
        return None
    return completion_tokens


def _optional_int_tuple(value: object, name: str) -> tuple[int, ...] | None:
    """Converts a JSON list of ids to a tuple, passing ``None`` through."""
    if value is None:
        return None
    if not isinstance(value, list):
        raise RuntimeError(f"{name} must be a list of integers, got {value!r}")
    ids: list[int] = []
    for position, element in enumerate(value):
        if isinstance(element, bool) or not isinstance(element, int):
            raise RuntimeError(
                f"{name}[{position}] must be an integer, got {element!r}"
            )
        ids.append(element)
    return tuple(ids)


def _as_object_mapping(value: object, name: str) -> Mapping[str, object]:
    """Narrows a decoded JSON value to an object, or raises."""
    if not isinstance(value, dict):
        raise RuntimeError(
            f"{name} must be a JSON object, got {type(value).__name__}"
        )
    return value


def _root(base_url: str) -> str:
    """Returns ``base_url`` without a trailing slash, ready for appending."""
    return base_url.rstrip("/")
