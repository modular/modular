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
"""Component-test reproducer for CENG-640.

These tests pin down the preprocessor performance fixes for image/video
downloads without an LLM in the loop:

* base64 ``data:`` decoding is offloaded to a worker thread (does not block the
  event loop) for large payloads, and runs inline for small ones;
* media size is bounded in aggregate by ``max_media_bytes``: an ``http(s)``
  download is aborted via the advertised ``Content-Length`` or a streamed-total
  guard once the request's cumulative media crosses that limit;
* the OpenAI route enforces the per-request video count up front, mirroring the
  image count cap.

The headline reproducer is ``test_event_loop_not_blocked_during_decode``: it
deadlocks (and fails) if the decode runs on the event loop, and passes only
when the decode is offloaded -- exactly the regression behind the high TTFT.
"""

from __future__ import annotations

import asyncio
import base64
import collections
import io
import logging
import threading
from collections.abc import AsyncIterator
from typing import Any
from unittest import mock

import pytest
from httpx import ConnectError, ReadTimeout
from max.pipelines.context.exceptions import InputError
from max.serve.config import Settings
from max.serve.pipelines.preprocess_cache_stats import (
    preprocessed_image_probe,
)
from max.serve.router import _image_resolution
from max.serve.router._image_resolution import (
    _METRIC_IMAGE_FORMATS,
    make_media_ref,
    resolve_image_from_url,
)
from max.serve.router.openai_routes import (
    openai_parse_chat_completion_request,
)
from max.serve.schemas.openai import CreateChatCompletionRequest
from PIL import Image
from pydantic import AnyUrl

pytestmark = pytest.mark.asyncio


def _data_uri(payload: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(payload).decode()


def _png_bytes(size: tuple[int, int] = (8, 8)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color="blue").save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Fix 2: base64 decode runs off the event loop for large payloads.
# ---------------------------------------------------------------------------


async def test_large_data_uri_decode_runs_off_event_loop(monkeypatch) -> None:  # noqa: ANN001
    """A large ``data:`` payload is decoded on a worker thread, not the loop."""
    main_thread = threading.get_ident()
    recorded: dict[str, int] = {}
    original = _image_resolution._decode_base64

    def spy(b64: str) -> bytes:
        recorded["thread"] = threading.get_ident()
        return original(b64)

    monkeypatch.setattr(_image_resolution, "_decode_base64", spy)

    # >256KiB of base64 -> exceeds the offload threshold.
    payload = b"\x00" * (400 * 1024)
    out = await resolve_image_from_url(
        AnyUrl(_data_uri(payload)), settings=Settings()
    )
    assert out == payload
    assert recorded["thread"] != main_thread


async def test_small_data_uri_decode_runs_inline(monkeypatch) -> None:  # noqa: ANN001
    """A tiny ``data:`` payload decodes inline (no thread-pool hop)."""
    main_thread = threading.get_ident()
    recorded: dict[str, int] = {}
    original = _image_resolution._decode_base64

    def spy(b64: str) -> bytes:
        recorded["thread"] = threading.get_ident()
        return original(b64)

    monkeypatch.setattr(_image_resolution, "_decode_base64", spy)

    payload = b"\x01" * 512
    out = await resolve_image_from_url(
        AnyUrl(_data_uri(payload)), settings=Settings()
    )
    assert out == payload
    assert recorded["thread"] == main_thread


async def test_event_loop_not_blocked_during_decode(monkeypatch) -> None:  # noqa: ANN001
    """Headline reproducer: the event loop stays responsive during decode.

    The decode is made to block on an :class:`threading.Event` that can only be
    released by a coroutine running concurrently on the event loop. If the
    decode ran *on* the loop (the CENG-640 regression), the releaser could never
    run and this would dead-lock until the timeout -> test failure. It passes
    only because the decode is offloaded to a worker thread, leaving the loop
    free to make progress.
    """
    started = threading.Event()
    release = threading.Event()
    original = _image_resolution._decode_base64

    def blocking_decode(b64: str) -> bytes:
        started.set()
        if not release.wait(timeout=5.0):
            raise AssertionError(
                "decode was never released: the event loop was blocked"
            )
        return original(b64)

    monkeypatch.setattr(_image_resolution, "_decode_base64", blocking_decode)

    payload = b"\x00" * (400 * 1024)

    async def releaser() -> None:
        # Runs on the same event loop as resolve. It can only make progress if
        # the loop is free while the decode is in flight.
        while not started.is_set():
            await asyncio.sleep(0.001)
        release.set()

    resolve_task = asyncio.create_task(
        resolve_image_from_url(AnyUrl(_data_uri(payload)), settings=Settings())
    )
    out, _ = await asyncio.wait_for(
        asyncio.gather(resolve_task, releaser()), timeout=15.0
    )
    assert out == payload


# ---------------------------------------------------------------------------
# Fix 1: reject oversized media before download/decode.
# ---------------------------------------------------------------------------


async def test_oversized_data_uri_rejected() -> None:
    """A ``data:`` payload over the request media budget is rejected (400).

    (Data URIs are also bounded by the body-size middleware, which caps the
    whole request body under its own ``max_request_bytes``; this exercises the
    resolver's aggregate media charge directly.)
    """
    payload = b"\x00" * 4096  # decodes to 4096 bytes, budget is 1024
    with pytest.raises(InputError, match="exceeds the maximum media size"):
        await resolve_image_from_url(
            AnyUrl(_data_uri(payload)),
            settings=Settings(max_media_bytes=1024),
        )


async def test_data_uri_within_budget_roundtrips() -> None:
    """A within-budget ``data:`` payload resolves to the exact original bytes."""
    payload = bytes(range(256)) * 8
    out = await resolve_image_from_url(
        AnyUrl(_data_uri(payload)), settings=Settings()
    )
    assert out == payload


# ---------------------------------------------------------------------------
# Fix 1: http(s) early-abort on size, via a fake streaming client.
# ---------------------------------------------------------------------------


def _no_ssrf(**overrides: Any) -> Settings:
    # These fetch tests drive the fake ``client.stream(...)`` directly, so they
    # exercise the break-glass (unvalidated) streaming path rather than the
    # SSRF-guarded path that would resolve the host through real DNS.
    return Settings(media_url_ssrf_protection_enabled=False, **overrides)


class _FakeResponse:
    def __init__(
        self,
        *,
        headers: dict[str, str],
        chunks: list[bytes],
        read_log: list[int],
        status_code: int = 200,
    ) -> None:
        self.headers = headers
        self.status_code = status_code
        self._chunks = chunks
        self._read_log = read_log

    def raise_for_status(self) -> None:
        return None

    async def aiter_bytes(
        self, chunk_size: int | None = None
    ) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            self._read_log.append(len(chunk))
            yield chunk


class _FakeStream:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    async def __aenter__(self) -> _FakeResponse:
        return self._response

    async def __aexit__(self, *exc: object) -> bool:
        return False


class _FakeAsyncClient:
    def __init__(self, response: _FakeResponse, **_: Any) -> None:
        self._response = response

    async def __aenter__(self) -> _FakeAsyncClient:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    def stream(self, method: str, url: str, **_: Any) -> _FakeStream:
        return _FakeStream(self._response)


def _install_fake_client(
    monkeypatch,  # noqa: ANN001
    *,
    headers: dict[str, str],
    chunks: list[bytes],
) -> list[int]:
    read_log: list[int] = []
    response = _FakeResponse(headers=headers, chunks=chunks, read_log=read_log)
    monkeypatch.setattr(
        _image_resolution,
        "AsyncClient",
        lambda **kw: _FakeAsyncClient(response),
    )
    return read_log


async def test_http_oversized_content_length_rejected_without_download(
    monkeypatch,  # noqa: ANN001
) -> None:
    """An over-budget advertised ``Content-Length`` rejects before any body read."""
    read_log = _install_fake_client(
        monkeypatch,
        headers={"content-length": str(100 * 1024 * 1024)},
        chunks=[b"x" * 1024],
    )
    with pytest.raises(InputError, match="exceeds the maximum media size"):
        await resolve_image_from_url(
            AnyUrl("https://example.com/big.mp4"),
            settings=_no_ssrf(max_media_bytes=50 * 1024 * 1024),
            media_kind="video",
        )
    # Body was never streamed.
    assert read_log == []


async def test_http_stream_aborts_when_total_exceeds_cap(
    monkeypatch,  # noqa: ANN001
) -> None:
    """With no/short Content-Length, the stream aborts once the total is over."""
    # Ten 60-byte chunks (600 bytes total), budget is 100 bytes; only the first
    # two chunks should be read before the abort.
    read_log = _install_fake_client(
        monkeypatch,
        headers={},  # no content-length advertised
        chunks=[b"a" * 60 for _ in range(10)],
    )
    with pytest.raises(InputError, match="exceeds the maximum media size"):
        await resolve_image_from_url(
            AnyUrl("https://example.com/sneaky.png"),
            settings=_no_ssrf(max_media_bytes=100),
        )
    assert len(read_log) == 2  # aborted early, not all ten chunks


async def test_http_within_cap_downloads_fully(monkeypatch) -> None:  # noqa: ANN001
    """A within-cap http download returns the concatenated body."""
    read_log = _install_fake_client(
        monkeypatch,
        headers={"content-length": "12"},
        chunks=[b"abcd", b"efgh", b"ijkl"],
    )
    out = await resolve_image_from_url(
        AnyUrl("https://example.com/ok.png"),
        settings=_no_ssrf(),
    )
    assert out == b"abcdefghijkl"
    assert len(read_log) == 3


# ---------------------------------------------------------------------------
# Fix (PERF-2725): explicit fetch timeout + graceful timeout/transport handling.
# A slow/stalled or failed download must raise a clean InputError (-> 400), not
# an opaque ReadTimeout that surfaces as a 500 the client then retries.
# ---------------------------------------------------------------------------


class _TimeoutResponse(_FakeResponse):
    """Streams one chunk, then raises ``ReadTimeout`` like a stalled download."""

    async def aiter_bytes(
        self, chunk_size: int | None = None
    ) -> AsyncIterator[bytes]:
        self._read_log.append(7)
        yield b"partial"
        raise ReadTimeout("simulated mid-stream stall")


class _RaisingStream:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def __aenter__(self) -> _FakeResponse:
        raise self._exc

    async def __aexit__(self, *exc: object) -> bool:
        return False


class _RaisingClient:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def __aenter__(self) -> _RaisingClient:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    def stream(self, method: str, url: str, **_: Any) -> _RaisingStream:
        return _RaisingStream(self._exc)


async def test_http_read_timeout_raises_clean_input_error(
    monkeypatch,  # noqa: ANN001
) -> None:
    """A mid-stream stall (ReadTimeout) -> clean InputError, not an opaque 500."""
    response = _TimeoutResponse(headers={}, chunks=[], read_log=[])
    monkeypatch.setattr(
        _image_resolution,
        "AsyncClient",
        lambda **kw: _FakeAsyncClient(response),
    )
    with pytest.raises(InputError, match="timed out fetching video"):
        await resolve_image_from_url(
            AnyUrl("https://example.com/slow.mp4"),
            settings=_no_ssrf(),
            media_kind="video",
        )


async def test_http_transport_error_raises_clean_input_error(
    monkeypatch,  # noqa: ANN001
) -> None:
    """A connect/transport failure -> clean InputError, not a 500."""
    monkeypatch.setattr(
        _image_resolution,
        "AsyncClient",
        lambda **kw: _RaisingClient(ConnectError("simulated connect failure")),
    )
    with pytest.raises(InputError, match="failed to fetch video"):
        await resolve_image_from_url(
            AnyUrl("https://example.com/unreachable.mp4"),
            settings=_no_ssrf(),
            media_kind="video",
        )


async def test_http_client_uses_explicit_non_default_timeout(
    monkeypatch,  # noqa: ANN001
) -> None:
    """The fetch client is built with an explicit timeout > httpx's 5s default.

    The default httpx timeout (5s on every op) is too aggressive for media
    downloads; this pins that an explicit, more generous read timeout is set.
    """
    captured: dict[str, Any] = {}

    def _factory(**kw: Any) -> _FakeAsyncClient:
        captured.update(kw)
        return _FakeAsyncClient(
            _FakeResponse(
                headers={"content-length": "2"},
                chunks=[b"ok"],
                read_log=[],
            )
        )

    monkeypatch.setattr(_image_resolution, "AsyncClient", _factory)
    await resolve_image_from_url(
        AnyUrl("https://example.com/ok.png"),
        settings=_no_ssrf(),
    )
    assert "timeout" in captured, (
        "fetch client must be given an explicit timeout"
    )
    read_timeout = getattr(captured["timeout"], "read", None)
    assert read_timeout is not None and read_timeout > 5.0


# ---------------------------------------------------------------------------
# Fix 3: the OpenAI route enforces video count + byte caps up front.
# ---------------------------------------------------------------------------


def _video_request(urls: list[str]) -> CreateChatCompletionRequest:
    content: list[dict[str, Any]] = [{"type": "text", "text": "describe"}]
    content += [{"type": "video_url", "video_url": {"url": u}} for u in urls]
    return CreateChatCompletionRequest.model_validate(
        {"model": "test", "messages": [{"role": "user", "content": content}]}
    )


async def test_parse_rejects_too_many_videos_before_download(
    monkeypatch,  # noqa: ANN001
) -> None:
    """Over the per-request video count -> 400 without resolving any video."""
    resolve_calls = 0
    original = _image_resolution.resolve_image_from_url

    async def spy(*args: Any, **kwargs: Any) -> bytes:
        nonlocal resolve_calls
        resolve_calls += 1
        return await original(*args, **kwargs)

    monkeypatch.setattr(
        "max.serve.router.openai_routes.resolve_image_from_url", spy
    )

    request = _video_request([_data_uri(b"x" * 16) for _ in range(5)])
    with pytest.raises(InputError, match="too many videos"):
        await openai_parse_chat_completion_request(
            request,
            wrap_content=True,
            settings=Settings(),
            max_videos_per_request=3,
        )
    assert resolve_calls == 0


async def test_parse_rejects_oversized_video() -> None:
    """A video data URI over the request media budget -> 400 (video message)."""
    request = _video_request([_data_uri(b"\x00" * 8192)])
    with pytest.raises(InputError, match="video media exceeds the maximum"):
        await openai_parse_chat_completion_request(
            request,
            wrap_content=True,
            settings=Settings(max_media_bytes=1024),
        )


async def test_parse_accepts_within_cap_image_and_video() -> None:
    """A small image + small video within all caps parse successfully."""
    png = _png_bytes()
    image_uri = _data_uri(png)
    video_uri = _data_uri(b"\x20" * 64)
    request = CreateChatCompletionRequest.model_validate(
        {
            "model": "test",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hi"},
                        {"type": "image_url", "image_url": {"url": image_uri}},
                        {"type": "video_url", "video_url": {"url": video_uri}},
                    ],
                }
            ],
        }
    )
    parsed = await openai_parse_chat_completion_request(
        request,
        wrap_content=True,
        settings=Settings(),
        max_images_per_request=200,
        max_videos_per_request=20,
    )
    assert len(parsed.images) == 1
    assert len(parsed.videos) == 1
    assert parsed.images[0] == png
    assert parsed.videos[0] == b"\x20" * 64


# ---------------------------------------------------------------------------
# Settings: aggregate max_media_bytes media budget and default media_kind.
# ---------------------------------------------------------------------------


def test_settings_media_defaults() -> None:
    """The media label defaults to 'image'."""
    settings = Settings()
    assert settings.media_kind == "image"


async def test_request_budget_bounds_media() -> None:
    """``Settings.max_media_bytes`` bounds resolved media in aggregate."""
    settings = Settings(max_media_bytes=1024)
    with pytest.raises(InputError, match="exceeds the maximum media size"):
        await resolve_image_from_url(
            AnyUrl(_data_uri(b"\x00" * 4096)), settings=settings
        )
    # Within the request budget -> resolves fine.
    out = await resolve_image_from_url(
        AnyUrl(_data_uri(b"\x00" * 512)), settings=settings
    )
    assert out == b"\x00" * 512


async def test_request_budget_is_shared_across_media() -> None:
    """One budget spans every item: two half-budget items together overrun it."""
    settings = Settings(max_media_bytes=1024)
    budget = _image_resolution._request_media_budget(settings)
    # First 600-byte item fits (600 <= 1024)...
    out = await resolve_image_from_url(
        AnyUrl(_data_uri(b"\x00" * 600)), settings=settings, budget=budget
    )
    assert out == b"\x00" * 600
    # ...but a second 600-byte item pushes the shared total to 1200 > 1024.
    with pytest.raises(InputError, match="exceeds the maximum media size"):
        await resolve_image_from_url(
            AnyUrl(_data_uri(b"\x00" * 600)), settings=settings, budget=budget
        )


async def test_settings_media_kind_used_in_error_message() -> None:
    """``Settings.media_kind`` labels the error when the caller omits it."""
    settings = Settings(max_media_bytes=1024, media_kind="video")
    with pytest.raises(InputError, match="video media exceeds the maximum"):
        await resolve_image_from_url(
            AnyUrl(_data_uri(b"\x00" * 4096)), settings=settings
        )


# ---------------------------------------------------------------------------
# ENABLE-2953: the route skips the admission decode for images the tokenizer
# has already preprocessed.
# ---------------------------------------------------------------------------


def _image_request(images: list[bytes]) -> CreateChatCompletionRequest:
    content: list[dict[str, Any]] = [{"type": "text", "text": "describe"}]
    content += [
        {"type": "image_url", "image_url": {"url": _data_uri(image)}}
        for image in images
    ]
    return CreateChatCompletionRequest.model_validate(
        {"model": "test", "messages": [{"role": "user", "content": content}]}
    )


async def test_parse_skips_the_decode_for_a_cached_image() -> None:
    """A cached image's pixels are never touched at admission.

    This is the whole point: on a multi-turn computer-use conversation almost
    every image in a request is one an earlier turn already preprocessed, and
    the decode of those images was 18% of the API server's busy CPU.
    """
    cached, fresh = _png_bytes((8, 8)), _png_bytes((16, 16))

    def mask(images: list[bytes], messages: list[Any]) -> list[bool]:
        return [image == cached for image in images]

    parsed = await openai_parse_chat_completion_request(
        _image_request([cached, fresh]),
        wrap_content=True,
        settings=Settings(),
        preprocessed_image_mask=mask,
    )

    assert parsed.images == [cached, fresh]
    assert parsed.decoded_images[0] is None
    assert parsed.decoded_images[1] is not None
    assert parsed.decoded_images[1].size == (16, 16)


async def test_parse_decodes_everything_without_a_probe() -> None:
    """An architecture with no preprocessed-image cache is unaffected."""
    parsed = await openai_parse_chat_completion_request(
        _image_request([_png_bytes((8, 8)), _png_bytes((16, 16))]),
        wrap_content=True,
        settings=Settings(),
    )

    assert all(image is not None for image in parsed.decoded_images)


async def test_parse_still_rejects_a_bad_uncached_image() -> None:
    """Skipping cached images must not weaken validation of the rest."""
    cached = _png_bytes((8, 8))
    bad = b"definitely-not-an-image"

    def mask(images: list[bytes], messages: list[Any]) -> list[bool]:
        return [image == cached for image in images]

    with pytest.raises(InputError, match="invalid or unreadable"):
        await openai_parse_chat_completion_request(
            _image_request([cached, bad]),
            wrap_content=True,
            settings=Settings(),
            preprocessed_image_mask=mask,
        )


async def test_parse_decodes_everything_when_the_probe_length_is_wrong(
    caplog,  # noqa: ANN001
) -> None:
    """A malformed probe answer costs the optimization, not the request.

    The mask is positional; a short one used to index out of range and
    surface as a 500. Decoding every image is exactly the behaviour before
    the probe existed, so that is the fallback. (ENABLE-2953.)
    """

    def short_mask(images: list[bytes], messages: list[Any]) -> list[bool]:
        return [True]

    with caplog.at_level(logging.WARNING, logger="max.serve"):
        parsed = await openai_parse_chat_completion_request(
            _image_request([_png_bytes((8, 8)), _png_bytes((16, 16))]),
            wrap_content=True,
            settings=Settings(),
            preprocessed_image_mask=short_mask,
        )

    assert all(image is not None for image in parsed.decoded_images)
    assert "returned 1 entries for 2 image(s)" in caplog.text


async def test_parse_runs_the_probe_off_the_event_loop() -> None:
    """The probe shares the decode's worker hop, not the event loop.

    It hashes every request image, so it belongs off the loop with the other
    CPU-bound media work -- and in the same hop, since its answer is what
    decides what the decode may skip. A second ``to_thread`` would add a
    scheduling bounce to move ~0.084ms of hashing for a typical request.
    """
    loop_thread = threading.get_ident()
    probe_threads: list[int] = []

    def recording_probe(images: list[bytes], messages: list[Any]) -> list[bool]:
        probe_threads.append(threading.get_ident())
        return [False] * len(images)

    await openai_parse_chat_completion_request(
        _image_request([_png_bytes((8, 8)), _png_bytes((16, 16))]),
        wrap_content=True,
        settings=Settings(),
        preprocessed_image_mask=recording_probe,
    )

    assert probe_threads, "probe was never called"
    assert loop_thread not in probe_threads


def test_probe_resolution_requires_the_protocol() -> None:
    """Only a tokenizer implementing the protocol opts in."""

    class WithoutProbe:
        pass

    class WithProbe:
        def preprocessed_image_mask(
            self, images: list[bytes], messages: list[Any]
        ) -> list[bool]:
            return [False] * len(images)

    assert preprocessed_image_probe(WithoutProbe()) is None
    probe = preprocessed_image_probe(WithProbe())
    assert probe is not None
    assert probe([b"a"], []) == [False]


def test_probe_resolution_rejects_a_non_callable_attribute(
    caplog,  # noqa: ANN001
) -> None:
    """A protocol check proves the attribute exists, not that it is callable.

    An architecture defining this as a list or property would otherwise reach
    the call site and fail there, which the route reports as a 400 about the
    request body -- pointing the operator at the client's JSON for a
    server-side mistake.
    """

    class BadProbe:
        preprocessed_image_mask = [True, False]

    with caplog.at_level(logging.WARNING, logger="max.serve"):
        assert preprocessed_image_probe(BadProbe()) is None
    assert "not callable" in caplog.text


async def test_parse_reports_preprocess_cache_hits_and_misses(
    monkeypatch,  # noqa: ANN001
) -> None:
    """The stage that dominates API-server CPU has to be observable."""
    recorded: dict[str, list[float]] = {
        "hits": [],
        "misses": [],
        "decode_ms": [],
    }
    monkeypatch.setattr(
        "max.serve.router.openai_routes.METRICS.vision_preprocess_cache_hits",
        lambda n: recorded["hits"].append(n),
    )
    monkeypatch.setattr(
        "max.serve.router.openai_routes.METRICS.vision_preprocess_cache_misses",
        lambda n: recorded["misses"].append(n),
    )
    monkeypatch.setattr(
        "max.serve.router.openai_routes.METRICS.image_admission_decode_time",
        lambda ms: recorded["decode_ms"].append(ms),
    )

    cached, fresh = _png_bytes((8, 8)), _png_bytes((16, 16))

    def mask(images: list[bytes], messages: list[Any]) -> list[bool]:
        return [image == cached for image in images]

    await openai_parse_chat_completion_request(
        _image_request([cached, fresh]),
        wrap_content=True,
        settings=Settings(),
        preprocessed_image_mask=mask,
    )

    assert recorded["hits"] == [1]
    assert recorded["misses"] == [1]
    assert len(recorded["decode_ms"]) == 1
    assert recorded["decode_ms"][0] >= 0.0


# ---------------------------------------------------------------------------
# CLIN-1911: the ``maxserve.media.*`` family. The media fetch path was
# entirely untimed, so a request that spent seconds downloading an image was
# indistinguishable from one that spent them in prefill.
# ---------------------------------------------------------------------------


class _RawRef:
    """A media reference that skips pydantic, so a malformed URL can be tested.

    ``AnyUrl`` rejects the inputs the resolver's own URL validation exists to
    reject. From a client URL those never reach ``_validate_and_pin`` -- they
    are refused in ``make_media_ref``, which
    ``test_a_client_url_that_is_not_url_shaped_is_counted`` drives end to
    end. What this double reaches is the *other* way those checks run: a
    redirect ``Location``, which is server-controlled and arrives as a raw
    string.
    """

    def __init__(self, raw: str, scheme: str) -> None:
        self._raw = raw
        self.scheme = scheme

    def unicode_string(self) -> str:
        return self._raw

    def __str__(self) -> str:
        return self._raw


def _media_metrics(monkeypatch) -> mock.MagicMock:  # noqa: ANN001
    """Patch the media instruments at the resolver's call site."""
    recorder = mock.MagicMock()
    monkeypatch.setattr(_image_resolution, "METRICS", recorder)
    return recorder


def _route_metrics(monkeypatch) -> mock.MagicMock:  # noqa: ANN001
    """Patch the media instruments at the route's own call site.

    Per-image facts are not here: ``emit_media_facts`` is shared with the
    responses router and records from ``_image_resolution``, so those tests
    take ``_media_metrics`` instead.
    """
    recorder = mock.MagicMock()
    monkeypatch.setattr("max.serve.router.openai_routes.METRICS", recorder)
    return recorder


async def test_resolve_counts_one_inline_and_one_fetched_item(
    monkeypatch,  # noqa: ANN001
) -> None:
    """The two schemes a real client uses are distinguished by ``source``.

    Both arms share one measured window, so a fetch cannot be timed while an
    inline base64 decode is not -- which is the state this family replaces.
    """
    png = _png_bytes((8, 8))
    _install_fake_client(
        monkeypatch,
        headers={"content-length": str(len(png))},
        chunks=[png],
    )
    recorder = _media_metrics(monkeypatch)
    settings = _no_ssrf()

    inline = await resolve_image_from_url(
        AnyUrl(_data_uri(png)), settings=settings, media_kind="image"
    )
    fetched = await resolve_image_from_url(
        AnyUrl("https://example.com/a.png"),
        settings=settings,
        media_kind="image",
    )

    assert inline == png
    assert fetched == png
    assert collections.Counter(
        call.args for call in recorder.media_items.call_args_list
    ) == {("image", "inline"): 1, ("image", "url"): 1}
    assert [
        call.args[1] for call in recorder.media_resolve_time.call_args_list
    ] == ["inline", "url"]
    assert all(
        call.args[0] >= 0.0
        for call in recorder.media_resolve_time.call_args_list
    )
    assert [call.args for call in recorder.media_item_size.call_args_list] == [
        (len(png), "image"),
        (len(png), "image"),
    ]
    recorder.media_rejections.assert_not_called()


async def test_file_scheme_resolves_under_its_own_source(
    monkeypatch,  # noqa: ANN001
    tmp_path,  # noqa: ANN001
) -> None:
    """``file:`` is its own ``source``, not folded into inline or fetch."""
    png = _png_bytes((8, 8))
    path = tmp_path / "local.png"
    path.write_bytes(png)
    recorder = _media_metrics(monkeypatch)

    out = await resolve_image_from_url(
        AnyUrl(path.as_uri()),
        settings=Settings(allowed_image_roots=[str(tmp_path)]),
        media_kind="image",
    )

    assert out == png
    recorder.media_items.assert_called_once_with("image", "file")
    recorder.media_item_size.assert_called_once_with(len(png), "image")


async def test_over_budget_item_counts_only_a_rejection(
    monkeypatch,  # noqa: ANN001
) -> None:
    """A refused item is counted once, as a rejection, and nowhere else.

    Counting it under ``items`` as well would make the rejection rate read as
    a fraction of a denominator that already excludes the successes.
    """
    recorder = _media_metrics(monkeypatch)

    with pytest.raises(InputError, match="exceeds the maximum media size"):
        await resolve_image_from_url(
            AnyUrl(_data_uri(b"\x00" * 4096)),
            settings=Settings(max_media_bytes=1024),
            media_kind="image",
        )

    recorder.media_rejections.assert_called_once_with("over_budget")
    recorder.media_items.assert_not_called()
    recorder.media_resolve_time.assert_not_called()
    recorder.media_item_size.assert_not_called()


async def test_url_failures_land_in_their_own_rejection_buckets(
    monkeypatch,  # noqa: ANN001
) -> None:
    """Each URL failure mode is its own ``reason``, not one 'bad media' bucket.

    Operators act differently on a malformed URL (a client bug) than on a
    host with no DNS record (an unreachable source).
    """
    for raw, scheme, reason in (
        ("http://[::1", "http", "malformed_url"),
        ("http:///no-host.png", "http", "no_host"),
        ("ftp://example.com/a.png", "ftp", "bad_scheme"),
    ):
        recorder = _media_metrics(monkeypatch)
        with pytest.raises(ValueError):
            await resolve_image_from_url(
                _RawRef(raw, scheme), settings=Settings(), media_kind="image"
            )
        recorder.media_rejections.assert_called_once_with(reason)
        recorder.media_items.assert_not_called()


async def test_a_client_url_that_is_not_url_shaped_is_counted(
    monkeypatch,  # noqa: ANN001
) -> None:
    """A URL pydantic refuses is the most ordinary client bug of all.

    It used to raise a bare ``ValidationError`` out of ``make_media_ref``,
    which counts nothing -- so the refusal rate read low by exactly the
    failures an operator is most likely to be looking for.
    """
    recorder = _media_metrics(monkeypatch)

    with pytest.raises(InputError, match="malformed media URL"):
        make_media_ref("http://[::1")

    recorder.media_rejections.assert_called_once_with("malformed_url")


async def test_a_malformed_client_url_is_counted_through_the_route(
    monkeypatch,  # noqa: ANN001
) -> None:
    """Driven from the request body, not from the resolver's own entry."""
    recorder = _media_metrics(monkeypatch)
    request = CreateChatCompletionRequest.model_validate(
        {
            "model": "test",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "http://[::1"},
                        }
                    ],
                }
            ],
        }
    )

    with pytest.raises(InputError, match="malformed media URL"):
        await openai_parse_chat_completion_request(
            request, wrap_content=True, settings=Settings()
        )

    recorder.media_rejections.assert_called_once_with("malformed_url")


@pytest.mark.parametrize(
    "payload",
    [
        "data:image/png;base64,",
        "data:image/png;base64,!!!not base64!!!",
        # Past the offload threshold, so the raise happens in a worker
        # thread and the rejection has to be counted back on the event loop.
        "data:image/png;base64" + "A" * (300 * 1024),
    ],
    ids=["empty", "not-base64", "offloaded"],
)
async def test_an_undecodable_inline_payload_counts_a_rejection(
    monkeypatch,  # noqa: ANN001
    payload: str,
) -> None:
    """A corrupt base64 body is one of the commonest real client bugs.

    It used to escape as a bare ``ValueError``/``binascii.Error``, counted
    nowhere, on the one media path that never touches the network.
    """
    recorder = _media_metrics(monkeypatch)

    with pytest.raises(ValueError):
        await resolve_image_from_url(
            _RawRef(payload, "data"),
            settings=Settings(),
            media_kind="image",
        )

    recorder.media_rejections.assert_called_once_with("bad_data_uri")
    recorder.media_items.assert_not_called()


async def test_a_refused_file_uri_counts_a_rejection(
    monkeypatch,  # noqa: ANN001
    tmp_path,  # noqa: ANN001
) -> None:
    """``file:`` misconfiguration is exactly what this counter is reached for.

    Every refusal in the block raised bare, so an operator whose
    ``allowed_image_roots`` does not cover the path they configured saw a
    flat zero.
    """
    recorder = _media_metrics(monkeypatch)

    with pytest.raises(ValueError, match="outside allowed roots"):
        await resolve_image_from_url(
            _RawRef(f"file://{tmp_path}/outside.png", "file"),
            settings=Settings(allowed_image_roots=["/nonexistent-root"]),
            media_kind="image",
        )

    recorder.media_rejections.assert_called_once_with("bad_file_uri")
    recorder.media_items.assert_not_called()


async def test_fetch_timeout_counts_its_own_rejection(
    monkeypatch,  # noqa: ANN001
) -> None:
    """A stalled download is attributable to the source, not to the server."""
    monkeypatch.setattr(
        _image_resolution,
        "AsyncClient",
        lambda **kw: _RaisingClient(ReadTimeout("simulated stall")),
    )
    recorder = _media_metrics(monkeypatch)

    with pytest.raises(InputError, match="timed out fetching"):
        await resolve_image_from_url(
            AnyUrl("https://example.com/slow.png"),
            settings=_no_ssrf(),
            media_kind="image",
        )

    recorder.media_rejections.assert_called_once_with("fetch_timeout")
    recorder.media_items.assert_not_called()


async def test_parse_counts_media_items_per_admitted_request(
    monkeypatch,  # noqa: ANN001
) -> None:
    """Per-request counts are taken before the caps and the first byte."""
    recorder = _route_metrics(monkeypatch)
    request = CreateChatCompletionRequest.model_validate(
        {
            "model": "test",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hi"},
                        {
                            "type": "image_url",
                            "image_url": {"url": _data_uri(_png_bytes())},
                        },
                        {
                            "type": "video_url",
                            "video_url": {"url": _data_uri(b"\x20" * 64)},
                        },
                    ],
                }
            ],
        }
    )

    await openai_parse_chat_completion_request(
        request, wrap_content=True, settings=Settings()
    )

    assert collections.Counter(
        call.args for call in recorder.media_items_per_request.call_args_list
    ) == {(1, "image"): 1, (1, "video"): 1}
    recorder.media_rejections.assert_not_called()


async def test_parse_counts_an_over_count_request_as_a_rejection(
    monkeypatch,  # noqa: ANN001
) -> None:
    """Over the per-request image cap is a rejection, and still sampled.

    The right tail is what sizes the cap, so counting past the cap hid the
    one request class this distribution exists to show -- and put MAX and
    Mach on opposite semantics under the same metric name.
    """
    recorder = _route_metrics(monkeypatch)

    with pytest.raises(InputError, match="too many images"):
        await openai_parse_chat_completion_request(
            _image_request([_png_bytes(), _png_bytes((16, 16))]),
            wrap_content=True,
            settings=Settings(),
            max_images_per_request=1,
        )

    recorder.media_rejections.assert_called_once_with("too_many_images")
    recorder.media_items_per_request.assert_called_once_with(2, "image")


async def test_a_text_only_request_emits_nothing_from_the_media_family(
    monkeypatch,  # noqa: ANN001
) -> None:
    """The family must be silent on the overwhelming majority of traffic."""
    route = _route_metrics(monkeypatch)
    resolver = _media_metrics(monkeypatch)

    await openai_parse_chat_completion_request(
        CreateChatCompletionRequest.model_validate(
            {
                "model": "test",
                "messages": [{"role": "user", "content": "just text"}],
            }
        ),
        wrap_content=True,
        settings=Settings(),
    )

    route.media_items_per_request.assert_not_called()
    route.media_rejections.assert_not_called()
    resolver.media_items.assert_not_called()
    resolver.media_resolve_time.assert_not_called()
    resolver.media_item_size.assert_not_called()
    resolver.media_rejections.assert_not_called()


# ---------------------------------------------------------------------------
# CLIN-1911 child 5: per-image format, pixels and codec decode cost. The
# window is the codec alone -- the per-architecture processor runs downstream
# on an already-decoded buffer, and folding it in would rank codecs by the
# pixel distribution of the images that happen to use them.
# ---------------------------------------------------------------------------


def _mpo_bytes() -> bytes:
    """A multi-picture JPEG, which PIL reports as ``MPO``, not ``JPEG``."""
    buf = io.BytesIO()
    first = Image.new("RGB", (8, 8), color="red")
    second = Image.new("RGB", (8, 8), color="green")
    first.save(buf, format="MPO", save_all=True, append_images=[second])
    return buf.getvalue()


async def test_decode_reports_format_and_pixels_for_every_image(
    monkeypatch,  # noqa: ANN001
) -> None:
    """Per-image size and cost, both tagged with the codec that produced them."""
    recorder = _media_metrics(monkeypatch)

    await openai_parse_chat_completion_request(
        _image_request([_png_bytes((8, 8)), _png_bytes((16, 16))]),
        wrap_content=True,
        settings=Settings(),
    )

    assert [call.args for call in recorder.media_image_size.call_args_list] == [
        (64, "png"),
        (256, "png"),
    ]
    assert [
        call.args for call in recorder.media_image_decodes.call_args_list
    ] == [("png",), ("png",)]
    decode_ms = recorder.media_image_decode_ms.call_args_list
    assert [call.args[1] for call in decode_ms] == ["png", "png"]
    assert all(call.args[0] > 0.0 for call in decode_ms)
    recorder.media_rejections.assert_not_called()


async def test_a_multi_picture_jpeg_reports_the_mpo_format(
    monkeypatch,  # noqa: ANN001
) -> None:
    """PIL names a multi-picture JPEG ``MPO``, so the label set has to hold it.

    Without it, every dual-camera phone photo would land in ``unknown`` and
    the per-codec panel would show a large unattributable bucket.
    """
    recorder = _media_metrics(monkeypatch)

    await openai_parse_chat_completion_request(
        _image_request([_mpo_bytes()]),
        wrap_content=True,
        settings=Settings(),
    )

    recorder.media_image_size.assert_called_once_with(64, "mpo")
    assert "mpo" in _METRIC_IMAGE_FORMATS


async def test_every_reported_format_comes_from_the_bounded_set(
    monkeypatch,  # noqa: ANN001
) -> None:
    """The label is not attacker-controllable: it is an allow-list or unknown."""
    recorder = _media_metrics(monkeypatch)

    await openai_parse_chat_completion_request(
        _image_request([_png_bytes(), _mpo_bytes()]),
        wrap_content=True,
        settings=Settings(),
    )

    reported = {
        call.args[1] for call in recorder.media_image_size.call_args_list
    }
    assert reported
    assert reported <= _METRIC_IMAGE_FORMATS | {"unknown"}


async def test_a_cache_skipped_image_reports_size_but_no_decode_counts(
    monkeypatch,  # noqa: ANN001
) -> None:
    """A skipped image is still sized; only its decode cost is absent.

    The skip is decided after the header read, so dropping format and pixels
    for it would bias the size distribution toward whatever the tokenizer
    has not seen before.
    """
    recorder = _media_metrics(monkeypatch)
    cached, fresh = _png_bytes((8, 8)), _png_bytes((16, 16))

    def mask(images: list[bytes], messages: list[Any]) -> list[bool]:
        return [image == cached for image in images]

    await openai_parse_chat_completion_request(
        _image_request([cached, fresh]),
        wrap_content=True,
        settings=Settings(),
        preprocessed_image_mask=mask,
    )

    assert [call.args for call in recorder.media_image_size.call_args_list] == [
        (64, "png"),
        (256, "png"),
    ]
    # Only the image that actually ran load() is counted and timed.
    assert [
        call.args for call in recorder.media_image_decodes.call_args_list
    ] == [("png",)]
    (decode_ms,) = recorder.media_image_decode_ms.call_args_list
    assert decode_ms.args[1] == "png"
    assert decode_ms.args[0] > 0.0


async def test_an_undecodable_image_still_reports_the_ones_before_it(
    monkeypatch,  # noqa: ANN001
) -> None:
    """The partial report is the point of the out parameter.

    The decode raises out of ``asyncio.to_thread`` with no return value, so a
    list owned by the decoding frame would be lost with it. This asserts
    image 1's format, pixels and decode time survive image 2's rejection.
    """
    recorder = _media_metrics(monkeypatch)

    with pytest.raises(InputError, match="invalid or unreadable"):
        await openai_parse_chat_completion_request(
            _image_request([_png_bytes((8, 8)), b"not an image at all"]),
            wrap_content=True,
            settings=Settings(),
        )

    recorder.media_image_size.assert_called_once_with(64, "png")
    recorder.media_image_decodes.assert_called_once_with("png")
    (decode_ms,) = recorder.media_image_decode_ms.call_args_list
    assert decode_ms.args[1] == "png"
    assert decode_ms.args[0] > 0.0
    recorder.media_rejections.assert_called_once_with("undecodable")


async def test_an_oversized_image_reports_its_own_rejection_reason(
    monkeypatch,  # noqa: ANN001
) -> None:
    """A decompression bomb is refused from its header, before the allocation.

    Its format and pixels are reported -- that is the evidence for tuning the
    limit -- under a reason distinct from an unreadable payload.
    """
    recorder = _media_metrics(monkeypatch)

    with pytest.raises(InputError, match="image decodes to"):
        await openai_parse_chat_completion_request(
            _image_request([_png_bytes((256, 256))]),
            wrap_content=True,
            settings=Settings(max_media_bytes=4096),
        )

    recorder.media_rejections.assert_called_once_with("decode_too_large")
    recorder.media_image_size.assert_called_once_with(65536, "png")
    recorder.media_image_decodes.assert_not_called()
    recorder.media_image_decode_ms.assert_not_called()


async def test_a_text_only_request_reports_no_per_image_facts(
    monkeypatch,  # noqa: ANN001
) -> None:
    recorder = _media_metrics(monkeypatch)

    await openai_parse_chat_completion_request(
        CreateChatCompletionRequest.model_validate(
            {
                "model": "test",
                "messages": [{"role": "user", "content": "just text"}],
            }
        ),
        wrap_content=True,
        settings=Settings(),
    )

    recorder.media_image_size.assert_not_called()
    recorder.media_image_decodes.assert_not_called()
    recorder.media_image_decode_ms.assert_not_called()


async def test_a_non_input_error_still_reports_the_images_before_it(
    monkeypatch,  # noqa: ANN001
) -> None:
    """``image.load()`` can raise ``MemoryError``, which is not an InputError.

    An 8-image request that OOMs on image 6 used to lose images 1-5's
    format, pixels and decode time -- exactly the evidence that failure
    calls for.
    """
    recorder = _media_metrics(monkeypatch)
    real_decode = _image_resolution.decode_and_validate_images

    def oom_on_the_second(
        images: list[bytes],
        max_decoded_bytes: int | None = None,
        skip_decode: Any = None,
        *,
        facts: Any = None,
    ) -> Any:
        real_decode(images[:1], max_decoded_bytes, None, facts=facts)
        raise MemoryError("simulated allocation failure")

    monkeypatch.setattr(
        "max.serve.router.openai_routes.decode_and_validate_images",
        oom_on_the_second,
    )

    with pytest.raises(MemoryError):
        await openai_parse_chat_completion_request(
            _image_request([_png_bytes((8, 8)), _png_bytes((16, 16))]),
            wrap_content=True,
            settings=Settings(),
        )

    recorder.media_image_size.assert_called_once_with(64, "png")
    recorder.media_image_decodes.assert_called_once_with("png")
