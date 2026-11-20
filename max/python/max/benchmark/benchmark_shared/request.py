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

"""Request-related data structures for benchmarking."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import os
import sys
import threading
import time
import traceback
from abc import ABC, abstractmethod
from array import array
from collections.abc import (
    AsyncIterator,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    Sequence,
)
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any, cast

import aiohttp
import msgspec
from max.serve._tool_call_validation import (
    check_response_format_conformance,
    response_format_schema_is_checkable,
)
from openai.types.chat.completion_create_params import ResponseFormat
from tqdm.asyncio import tqdm
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing_extensions import NotRequired, TypedDict

from .config import (
    PIXEL_GENERATION_TASKS,
    Backend,
    BenchmarkTask,
    SamplingConfig,
    get_video_request_schema,
)
from .datasets.types import (
    ChatMessage,
    ImageContentBlock,
    OpenAIImage,
    PixelGenerationImageOptions,
)
from .sse import iter_events
from .utils import deadline_passed, openai_bearer_auth_headers

# 30 minute timeout per request session
AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=30 * 60)

# How long a pooled connection may sit idle before it is closed. Long enough
# that a run's own think time rarely costs a handshake, short enough that a
# server which closes idle connections first does not hand back a dead one: a
# POST is not retried, so a race there fails the request outright.
#
# The native pool lives for one benchmark() point, while the measuring proxy's
# lives for the proxy, so the two only diverge with --skip-test-prompt and no
# warmup, where the proxy serves requests the native client never sends.
# TODO: the eval-capture proxy (utils/eval-capture, proxy.rs) still opens a
# connection per request; give it the same pool.
KEEPALIVE_TIMEOUT_S = 60

# Chunks tokenized per call when filling in per-chunk TPOT; bounds peak memory.
_TPOT_TOKENIZE_SLICE = 10_000

logger = logging.getLogger(__name__)


@contextlib.asynccontextmanager
async def benchmark_client_session() -> AsyncIterator[aiohttp.ClientSession]:
    """Yields the one keep-alive connection pool a benchmark run shares."""
    connector = aiohttp.TCPConnector(
        limit=0, limit_per_host=0, keepalive_timeout=KEEPALIVE_TIMEOUT_S
    )
    async with aiohttp.ClientSession(
        connector=connector, timeout=AIOHTTP_TIMEOUT
    ) as session:
        yield session


def _encode_openresponses_image_from_file_path(
    file_path: str,
) -> dict[str, str]:
    extension_to_mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }

    _, ext = os.path.splitext(file_path.lower())
    if ext not in extension_to_mime:
        supported_exts = ", ".join(extension_to_mime.keys())
        raise ValueError(
            f"Unsupported image file extension '{ext}'. "
            f"Supported extensions: {supported_exts}"
        )

    with open(file_path, "rb") as f:
        image_bytes = f.read()

    img_base64 = base64.b64encode(image_bytes).decode("utf-8")
    return {
        "type": "input_image",
        "image_url": f"data:{extension_to_mime[ext]};base64,{img_base64}",
    }


@dataclass
class BaseRequestFuncInput(ABC):
    """Base class for request function input with common fields."""

    model: str
    session_id: str | None
    # kw_only so this can default without disturbing the required-field
    # ordering of subclasses (RequestFuncInput adds several fields with no
    # default after inheriting from this base).
    cache_salt: str | None = field(default=None, kw_only=True)

    @abstractmethod
    def get_output_type(self) -> type[BaseRequestFuncOutput]:
        """Get the output type for the request function input."""


# The payload fields a ``SamplingConfig`` contributes: one scalar per set field.
SamplingFields = Mapping[str, float | int]
# The four ``SamplingConfig`` values a cached ``SamplingFields`` was built from.
SamplingFieldsKey = tuple[float | None, float | None, float | None, int | None]


def _apply_sampling_to_request_payload(
    payload: dict[str, Any], sampling: SamplingConfig
) -> None:
    """Merge non-None OpenAI-style sampling fields from *sampling* into *payload*."""
    if sampling.temperature is not None:
        payload["temperature"] = sampling.temperature
    if sampling.thinking_temperature is not None:
        payload["thinking_temperature"] = sampling.thinking_temperature
    if sampling.top_k is not None:
        payload["top_k"] = sampling.top_k
    if sampling.top_p is not None:
        payload["top_p"] = sampling.top_p


# A ChatMessage after model_dump(): "role" plus "content", which is either a
# plain string or a list of content blocks. Not a TypedDict because these
# arrive from model_dump() as dict[str, Any], which would need a cast.
SerializedChatMessage = MutableMapping[str, Any]


class ChatWireHistory:
    """Encodes each message once; the history only grows.

    A benchmark run appends one message per turn and re-sends the whole
    conversation each time, so each message must be serialized exactly once.
    The cache holds the array's elements and closes it per request.

    The cache cannot tell a rewritten message from a new one, because it never
    looks at a message it has already encoded: an edit to an earlier message
    goes out unchanged. Only a *shorter* history is caught, by ``encode``
    raising, and that error is meant to abort the run. A caller that dropped
    messages mid-run has broken the invariant that a session's prompt extends
    the previous one, so every later measurement would be attributed to the
    wrong request.
    """

    def __init__(self) -> None:
        # The messages encoded so far, as a JSON array missing its closer.
        self._encoded = bytearray(b"[")
        self._count = 0

    def encode(self, messages: Sequence[ChatMessage]) -> msgspec.Raw:
        """The JSON array of *messages*, ready to splice into a body.

        Raises:
            ValueError: If *messages* is shorter than what was already
                encoded. The cache only appends, so a shorter history means
                the caller dropped messages, and reusing the prefix would send
                them anyway.
        """
        if len(messages) < self._count:
            raise ValueError(
                f"Cannot encode {len(messages)} messages after "
                f"{self._count} were already sent: the cached history only "
                "grows."
            )
        for message in messages[self._count :]:
            if self._count:
                self._encoded += b", "
            self._encoded += msgspec.json.encode(message.model_dump())
            self._count += 1
        return msgspec.Raw(self._encoded + b"]")


def _attach_images_to_first_user_message(
    messages: Sequence[SerializedChatMessage],
    images: Sequence[OpenAIImage],
) -> bool:
    """Attach images to the first user message, normalizing string content.

    Neither "index 0" nor "content is a list" holds in general: a conversation
    can lead with a system message, and datasets such as agentic-code pass
    plain-string content straight through.

    The drop paths below are a safety net, not a supported outcome. Image
    augmentation skips prompts with no user message rather than counting
    tokens for images it knows would be dropped here, so reaching a drop
    means sampling and sending disagree about the prompt shape.

    Returns:
        Whether *images* reached the payload.
    """
    target = next((m for m in messages if m.get("role") == "user"), None)
    if target is None:
        logger.warning(
            "Dropping %d image(s): the conversation has no user message.",
            len(images),
        )
        return False
    content = target.get("content")
    if isinstance(content, str):
        target["content"] = [{"type": "text", "text": content}]
    elif not isinstance(content, list):
        logger.warning(
            "Dropping %d image(s): unexpected content type %s.",
            len(images),
            type(content).__name__,
        )
        return False
    target["content"].extend(images)
    return True


def prompt_carries_image(prompt: str | list[ChatMessage]) -> bool:
    """Whether *prompt*'s messages carry an image content part."""
    return isinstance(prompt, list) and any(
        isinstance(part, ImageContentBlock)
        for message in prompt
        if isinstance(content := message.content, list)
        for part in content
    )


def _chat_messages_json(
    prompt: str | list[ChatMessage],
    images: Sequence[OpenAIImage],
    history: ChatWireHistory | None,
) -> tuple[msgspec.Raw, bool]:
    """The ``messages`` array of a chat body, encoded.

    Returns:
        The array, and whether *images* were added to it.
    """
    if history is not None and not images and isinstance(prompt, list):
        return history.encode(prompt), False
    messages: list[SerializedChatMessage]
    if isinstance(prompt, str):
        messages = [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ]
    else:
        messages = [msg.model_dump() for msg in prompt]
    attached = bool(images) and _attach_images_to_first_user_message(
        messages, images
    )
    return msgspec.Raw(msgspec.json.encode(messages)), attached


def _build_final_payload(
    base_payload: Mapping[str, Any], extra_body: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Return a new payload with *extra_body* merged over *base_payload*.

    The copy is shallow and last-writer-wins (an *extra_body* key overrides
    the managed field of the same name). Not mutated, so *base_payload* keeps
    its precise type and the result is ``dict[str, Any]``.
    """
    payload: dict[str, Any] = dict(base_payload)
    if not extra_body:
        return payload
    for key, value in extra_body.items():
        if key in payload:
            logger.warning(
                "extra_body key %r overwrites managed request field; "
                "last-writer-wins.",
                key,
            )
        payload[key] = value
    return payload


@dataclass
class RequestFuncInput(BaseRequestFuncInput):
    """Request function input for text generation benchmarks."""

    sampling: SamplingConfig
    prompt: str | list[ChatMessage]
    images: list[OpenAIImage]
    api_url: str
    prompt_len: int
    max_tokens: int | None
    ignore_eos: bool
    response_format: ResponseFormat | None = None
    # Whether ``prompt`` already carries an image part from an earlier turn.
    prompt_carries_image: bool = False
    # Forwarded as the chat-completions ``tools`` field. Plain ``list[dict]``
    # so datasets can pass through OpenAI-shaped or server-specific schemas
    # without translation; ignored by non-chat drivers.
    tools: list[dict[str, Any]] | None = None
    # The adapter the request is routed to. ``model`` already names it for the
    # server; this keeps it apart from a base-model request for the metrics.
    lora_id: str | None = None
    # Cache seeding needs a completed generation, even if its tokens are
    # special tokens that the server does not expose as text.
    is_cache_warmup: bool = False
    # Wire form of ``prompt``'s messages already sent, for a growing history.
    wire_history: ChatWireHistory | None = field(
        default=None, repr=False, compare=False
    )

    def get_output_type(self) -> type[BaseRequestFuncOutput]:
        """Returns the output class this driver produces."""
        return RequestFuncOutput


@dataclass
class PixelGenerationRequestFuncInput(BaseRequestFuncInput):
    """Request function input for pixel-generation benchmarks."""

    prompt: str
    input_image_paths: list[str] | None
    api_url: str
    image_options: PixelGenerationImageOptions | None = None

    def get_output_type(self) -> type[BaseRequestFuncOutput]:
        """Returns the output class this driver produces."""
        return PixelGenerationRequestFuncOutput


@dataclass
class BaseRequestFuncOutput:
    """Base class for request function output with common fields."""

    cancelled: bool = False
    success: bool = False
    latency: float = 0.0
    error: str = ""
    # time.perf_counter() at request dispatch (monotonic, run-relative)
    request_submit_time: float | None = None

    @property
    def request_complete_time(self) -> float | None:
        """Derived completion timestamp: submit time + latency."""
        if self.request_submit_time is None:
            return None
        return self.request_submit_time + self.latency


def tag_lora_route(
    output: BaseRequestFuncOutput, request_func_input: BaseRequestFuncInput
) -> None:
    """Records the adapter *output*'s request was routed to, if any."""
    if isinstance(output, RequestFuncOutput) and isinstance(
        request_func_input, RequestFuncInput
    ):
        output.lora_id = request_func_input.lora_id


def tag_response_format_outcome(
    output: BaseRequestFuncOutput, request_func_input: BaseRequestFuncInput
) -> None:
    """Record whether *output* was constrained, and whether it conformed.

    Conformance is judged client-side, from the response text alone, so it
    reports on any backend rather than only on one that exposes a metrics
    endpoint.

    The check is the same one the server runs for its own conformance log, so
    the two numbers mean the same thing and can be compared. Failures here are
    a property of the response, never of the benchmark: the checker is
    documented as never raising.

    A schema that does not compile leaves the request unjudged rather than
    scoring it. The checker itself fails open, which is right for the server's
    log but would be backwards here -- an unusable schema would report a
    perfect conformance rate, and the metric reads healthiest exactly where it
    measured nothing.
    """
    if not isinstance(output, RequestFuncOutput) or not isinstance(
        request_func_input, RequestFuncInput
    ):
        return
    response_format = request_func_input.response_format
    output.response_format_constrained = response_format is not None
    if response_format is None:
        return
    schema = cast(dict[str, Any], response_format).get("json_schema")
    # json_object asks for any well-formed JSON object, which the permissive
    # schema below expresses; a bare "text" format constrains nothing at all.
    if isinstance(schema, dict):
        schema = schema.get("schema")
    elif cast(dict[str, Any], response_format).get("type") == "json_object":
        schema = {"type": "object"}
    else:
        return
    if not isinstance(schema, dict):
        return
    if not response_format_schema_is_checkable(schema):
        return
    result = check_response_format_conformance(output.generated_text, schema)
    output.response_format_conformed = result.outcome == "valid"


def mark_cancelled_if_past_deadline(
    output: BaseRequestFuncOutput, end_time_ns: int | None
) -> BaseRequestFuncOutput:
    """Reclassify an end-of-benchmark cut-off as cancelled rather than failed.

    When the benchmark duration deadline cancels an in-flight request, aiohttp
    can surface the cancellation as a swallowed error inside the request driver
    instead of propagating the ``wait_for`` timeout, leaving a ``success=False``
    output. If the result is not a success and the benchmark deadline has
    passed, treat it as cancelled (cut off by benchmark end) rather than a real
    failure. Successful results are left untouched.

    Args:
        output: The request output to (possibly) reclassify, mutated in place.
        end_time_ns: The benchmark ``perf_counter_ns`` deadline, or ``None`` if
            the run is unbounded.

    Returns:
        The same ``output`` instance, for convenient inline use.
    """
    if not output.success and deadline_passed(end_time_ns):
        output.cancelled = True
        logger.info(
            "Reclassifying request as cancelled (cut off by benchmark end): %s",
            output.error or "<no error>",
        )
    return output


def measured_window_duration(
    outputs: Iterable[BaseRequestFuncOutput], fallback: float
) -> float:
    """Wall-clock seconds from the first submit to the last complete.

    The window covers only requests with both a ``request_submit_time`` and a
    ``request_complete_time``. If no such request exists, return ``fallback``.
    Otherwise return ``max(last_complete - first_submit, 1e-9)`` so callers
    can safely divide.

    This is the same window math the steady-state block uses and is the
    correct denominator for aggregate throughput / TPM over a sliced benchmark
    region — warmup/tail wall time is excluded along with warmup/tail tokens.
    """
    first_submit: float | None = None
    last_complete: float | None = None
    for o in outputs:
        submit = o.request_submit_time
        if submit is None:
            continue
        complete = o.request_complete_time
        if complete is None:
            continue
        if first_submit is None or submit < first_submit:
            first_submit = submit
        if last_complete is None or complete > last_complete:
            last_complete = complete
    if first_submit is None or last_complete is None:
        return fallback
    return max(last_complete - first_submit, 1e-9)


@dataclass
class ServerTokenStats:
    """Server-reported token counts from the stream_options usage chunk."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cached_tokens: int = 0
    # Whether the usage carried cached-token counts at all, so a server that
    # reports none is not read as one that served nothing from its cache.
    cached_tokens_reported: bool = False


@dataclass
class RequestFuncOutput(BaseRequestFuncOutput):
    """Request function output for text generation benchmarks."""

    # List of inter-token latencies.
    itl: list[float] = field(default_factory=list)
    # List of per-chunk time-per-output-token values.
    tpot: list[float] = field(default_factory=list)
    # End offsets in ``generated_text``; compute_chunk_tpots drains it.
    pending_tpot_ends: array[int] = field(default_factory=lambda: array("I"))
    generated_text: str = ""
    ttft: float = 0.0  # Time to first token
    prompt_len: int = 0
    server_token_stats: ServerTokenStats = field(
        default_factory=ServerTokenStats
    )
    # Multi-turn provenance, set by the conversation driver so per-turn cache
    # retention can group/order turns within a session. None for single-turn.
    session_id: str | None = None
    turn_index: int | None = None
    # Whether this request carried a response_format. Set by the driver, which
    # is the last place holding both the input and the output -- metrics see
    # only the outputs.
    response_format_constrained: bool = False
    # Whether the generated text satisfied that response_format. ``None`` when
    # the request was unconstrained or produced nothing to judge.
    response_format_conformed: bool | None = None
    # Whether the request offered ``tools``, and whether the response contained
    # a tool call. Offered but never called means the prompts gave the model no
    # reason to use its tools: MAX Serve builds the tool grammar but never
    # enforces it, so the run measures the definitions in the prompt and little
    # of tool-call decoding.
    tools_offered: bool = False
    tool_call_returned: bool = False
    # Whether the payload carried an image part, counting images resent with
    # earlier turns of the session: the share production counts per request.
    carries_image: bool = False
    lora_id: str | None = None


@dataclass
class PixelGenerationRequestFuncOutput(BaseRequestFuncOutput):
    """Request function output for text-to-image benchmarks."""

    num_generated_outputs: int = 0


class RequestDriver(ABC):
    """Abstract base class for a driver that handles API requests to different backends."""

    def __init__(
        self,
        *,
        session: aiohttp.ClientSession,
        tokenizer: PreTrainedTokenizerBase | None = None,
        extra_body: Mapping[str, Any] | None = None,
        backend: Backend | None = None,
    ) -> None:
        """Initialize the request driver.

        Args:
            session: The client session whose connections this driver's
                requests share.
            tokenizer: Optional tokenizer for per-chunk TPOT.
            extra_body: Optional arbitrary top-level fields merged onto every
                request payload (last-writer-wins). Consumed by the
                text-generation drivers (chat completions, completions, and
                TensorRT-LLM); other drivers ignore it.
            backend: The inference backend. Used by
                :class:`OpenAIChatCompletionsRequestDriver` to enable the ATOM
                server-reported-timing workaround for the ``atom`` backend.
                TODO(ATOM): remove once ATOM streams chat/completions correctly.
        """
        self.session = session
        self.tokenizer = tokenizer
        self.extra_body = extra_body
        self.backend = backend
        # Per-run constants that the drivers would otherwise rebuild on every
        # request. The wrapper drivers below delegate and never read them.
        self._sampling_cache: (
            tuple[SamplingFieldsKey, SamplingFields] | None
        ) = None
        self._auth: Mapping[str, str] | None = None

    def _sampling_fields(self, sampling: SamplingConfig) -> SamplingFields:
        """The request payload fields *sampling* contributes, built once per run.

        Keyed on the four values, not the object: callers rebuild an equal
        ``SamplingConfig`` per request, so identity would miss every time, and
        holding the object itself would keep reporting the old fields after
        an in-place edit, because comparing it to itself always matches.
        """
        key = (
            sampling.temperature,
            sampling.thinking_temperature,
            sampling.top_p,
            sampling.top_k,
        )
        cached = self._sampling_cache
        if cached is not None and cached[0] == key:
            return cached[1]
        fields: dict[str, float | int] = {}
        _apply_sampling_to_request_payload(fields, sampling)
        self._sampling_cache = (key, fields)
        return fields

    def _auth_headers(self) -> Mapping[str, str]:
        """The run's bearer auth headers, read from the environment once."""
        if self._auth is None:
            self._auth = openai_bearer_auth_headers()
        return self._auth

    @abstractmethod
    async def request(
        self, request_func_input: BaseRequestFuncInput
    ) -> BaseRequestFuncOutput:
        """Execute a request to the backend API.

        Args:
            request_func_input: Input parameters for the request.

        Returns:
            RequestFuncOutput containing the response data and metrics.
        """


class ProgressBarRequestDriver(RequestDriver):
    """Request driver that updates a progress bar after each request."""

    def __init__(
        self,
        request_driver: RequestDriver,
        pbar: tqdm,
    ) -> None:
        """Initialize the progress bar request driver.

        Args:
            request_driver: The underlying request driver to wrap.
            pbar: Progress bar to update after each request completes.
        """
        super().__init__(
            session=request_driver.session, tokenizer=request_driver.tokenizer
        )
        self.request_driver = request_driver
        self.pbar = pbar

    async def request(
        self, request_func_input: BaseRequestFuncInput
    ) -> BaseRequestFuncOutput:
        """Execute a request to the backend API.

        Args:
            request_func_input: Input parameters for the request.

        Returns:
            RequestFuncOutput containing the response data and metrics.
        """
        result = await self.request_driver.request(request_func_input)
        self.pbar.update(1)
        return result


@contextlib.contextmanager
def progressbar_request_driver(
    request_driver: RequestDriver,
    total: int,
    *,
    disable_tqdm: bool = False,
    desc: str | None = None,
) -> Iterator[RequestDriver]:
    """Yield a request driver that advances a progress bar per request.

    When *disable_tqdm* is set, the driver is yielded unwrapped and no bar is
    shown. Otherwise the driver is wrapped in a :class:`ProgressBarRequestDriver`
    backed by a ``tqdm`` bar that is closed on exit.

    Args:
        request_driver: The underlying request driver to wrap.
        total: Total number of requests the bar tracks.
        disable_tqdm: If True, skip the progress bar entirely.
        desc: Optional description shown alongside the bar.

    Yields:
        The (possibly progress-wrapped) request driver.
    """
    if disable_tqdm:
        yield request_driver
        return
    pbar = tqdm(total=total, desc=desc)
    try:
        yield ProgressBarRequestDriver(request_driver, pbar)
    finally:
        pbar.close()


class TRTLLMRequestDriver(RequestDriver):
    """Request driver for TensorRT-LLM backend."""

    async def request(
        self, request_func_input: BaseRequestFuncInput
    ) -> RequestFuncOutput:
        """Execute a request to the TensorRT-LLM backend."""
        if not isinstance(request_func_input, RequestFuncInput):
            raise TypeError("TRTLLMRequestDriver requires RequestFuncInput.")
        api_url = request_func_input.api_url
        assert api_url.endswith("generate_stream")

        base_payload: dict[
            str, bool | str | int | float | list[ChatMessage]
        ] = {
            "text_input": request_func_input.prompt,
            "ignore_eos": request_func_input.ignore_eos,
            "stream": True,
        }

        if request_func_input.max_tokens is not None:
            base_payload["max_tokens"] = request_func_input.max_tokens
        _apply_sampling_to_request_payload(
            base_payload, request_func_input.sampling
        )
        payload = _build_final_payload(base_payload, self.extra_body)

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        ttft = 0.0
        chunk_texts: list[str] = []
        chunk_texts_chars = 0
        st = time.perf_counter()
        output.request_submit_time = st
        most_recent_timestamp = st
        try:
            async with self.session.post(url=api_url, json=payload) as response:
                if response.status == 200:
                    events = iter_events(response.content.iter_any())
                    try:
                        async for event in events:
                            chunk = _decode_chunk(event.data)
                            if not isinstance(chunk, Mapping):
                                raise _malformed_chunk(
                                    "expected a JSON object", event.data
                                )
                            chunk_text = _required_text_field(
                                chunk.get("text_output"),
                                "text_output",
                                event.data,
                            )
                            timestamp = time.perf_counter()
                            chunk_texts.append(chunk_text)
                            chunk_texts_chars += len(chunk_text)
                            if self.tokenizer is not None:
                                output.pending_tpot_ends.append(
                                    chunk_texts_chars
                                )
                            # First token
                            if ttft == 0.0:
                                ttft = timestamp - st
                                output.ttft = ttft

                            # Decoding phase
                            else:
                                itl_value = timestamp - most_recent_timestamp
                                output.itl.append(itl_value)

                            most_recent_timestamp = timestamp
                    finally:
                        await events.aclose()

                    output.latency = most_recent_timestamp - st
                    output.success = True

                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

        output.generated_text = "".join(chunk_texts)
        return output


def _append_slice_tpots(
    pending: Sequence[tuple[RequestFuncOutput, str, float]],
    tokenizer: PreTrainedTokenizerBase,
    failures: list[str],
) -> set[int]:
    """Tokenizes one slice of chunks and appends each chunk's TPOT.

    Args:
        pending: The slice's (response, chunk text, ITL) triples.
        tokenizer: The tokenizer that turns each chunk's text into tokens.
        failures: Receives a description of the first tokenizer failure, so the
            one warning per run can say what went wrong.

    Returns:
        The responses in the slice whose TPOT this slice dropped. A tokenizer
        that raises, or returns a different number of encodings than it was
        handed, leaves the whole slice unattributed; the affected responses
        also lose the TPOT of every earlier slice, so no response ends up with
        partial coverage.
    """
    try:
        encoded = tokenizer(
            [text for _, text, _ in pending], add_special_tokens=False
        )
        slice_tpots = [
            itl_value / len(token_ids) if token_ids else None
            for (_, _, itl_value), token_ids in zip(
                pending, encoded.input_ids, strict=True
            )
        ]
    except Exception as exc:
        # A tokenizer that rejects the generated text is the benchmark's
        # problem, not the request's: drop per-chunk TPOT for these responses
        # rather than failing the run's whole result over it.
        if not failures:
            failures.append(f"{type(exc).__name__}: {exc}")
        for output, _, _ in pending:
            output.tpot.clear()
        return {id(output) for output, _, _ in pending}
    for (output, _, _), tpot in zip(pending, slice_tpots, strict=True):
        if tpot is not None:
            output.tpot.append(tpot)
    return set()


def _malformed_chunk(why: str, data: str) -> TypeError:
    """Builds the client's error for a chunk it cannot read.

    The payload snippet is what makes the failure actionable: the shape came
    from the server, so whoever reads the log needs the bytes that broke it.
    """
    return TypeError(f"Malformed stream chunk: {why}: {data[:200]!r}")


def _decode_chunk(data: str) -> Any:
    """Decodes one event's data, quoting the payload if it is not JSON."""
    try:
        return msgspec.json.decode(data)
    except msgspec.DecodeError as exc:
        raise _malformed_chunk(f"not valid JSON ({exc})", data) from exc


def _text_field(value: Any, field: str, data: str) -> str:
    """Returns one streamed text field, reading an absent value as no text.

    Chat deltas may legitimately carry a null field, and a field we do not
    model is not an error. A field that is present with another type is
    rejected, because it would silently become missing text.
    """
    if value is None:
        return ""
    if not isinstance(value, str):
        raise _malformed_chunk(f"expected {field} to be a string", data)
    return value


def _required_text_field(value: Any, field: str, data: str) -> str:
    """Returns a text field the chunk must carry as a string.

    Used where a null would not mean "the server sent no text this chunk" but
    a schema the client does not understand: a completions ``choice.text`` is
    the response's text, not an optional delta field.
    """
    if not isinstance(value, str):
        raise _malformed_chunk(f"expected {field} to be a string", data)
    return value


def _server_token_stats(usage: Mapping[str, Any]) -> ServerTokenStats:
    """Reads one chunk's ``usage`` object into the run's token stats."""
    details = usage.get("prompt_tokens_details")
    cached = (
        details.get("cached_tokens") if isinstance(details, Mapping) else None
    )
    return ServerTokenStats(
        prompt_tokens=_usage_count(usage.get("prompt_tokens")),
        completion_tokens=_usage_count(usage.get("completion_tokens")),
        total_tokens=_usage_count(usage.get("total_tokens")),
        cached_tokens=_usage_count(cached) or 0,
        cached_tokens_reported=isinstance(details, Mapping),
    )


def _usage_count(value: Any) -> int | None:
    """Coerces one server-reported token count to an int.

    Providers are lax here, sending ``3``, ``3.0`` or ``"3"`` for the same
    count, and the counts feed integer arithmetic in the post-run metrics. A
    bool is rejected rather than coerced: ``true`` in a count field is a
    schema change, not a count of one.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"usage count is not an integer: {value!r}")
    if isinstance(value, int):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise TypeError(f"usage count is not an integer: {value!r}") from None
    if not number.is_integer():
        raise TypeError(f"usage count is not an integer: {value!r}")
    return int(number)


def compute_chunk_tpots(
    outputs: Iterable[RequestFuncOutput],
    tokenizer: PreTrainedTokenizerBase | None,
    *,
    disable_tqdm: bool | None = None,
) -> None:
    """Fills ``tpot`` from the chunk offsets recorded while running.

    Runs after the requests, so the event loop never tokenizes inside the
    measured window, one slice of text at a time.

    Note:
        The values are approximate: re-tokenizing the chunk text the client
        received need not match the server's own tokenization of what it
        generated.
    """
    outputs = list(outputs)
    if tokenizer is None or not any(
        output.success and len(output.pending_tpot_ends) > 1
        for output in outputs
    ):
        for output in outputs:
            output.pending_tpot_ends = array("I")
        return

    logger.info(f"Computing per-chunk TPOT for {len(outputs)} responses...")
    unsynced = 0  # Responses whose offsets and ITLs disagree.
    failed_slices = 0  # Tokenizer calls that left a slice unattributed.
    failures: list[str] = []  # The first such failure, for the one warning.
    dropped: set[int] = set()  # Responses a slice left without TPOT.
    pending: list[tuple[RequestFuncOutput, str, float]] = []
    with tqdm(desc="Per-chunk TPOT", unit="slice", disable=disable_tqdm) as bar:
        for output in outputs:
            offsets = output.pending_tpot_ends
            output.pending_tpot_ends = array("I")
            if not output.success or output.cancelled or not offsets:
                continue
            if len(offsets) - 1 != len(output.itl):
                # One offset pair and one ITL per streamed chunk, so this can
                # only mean the recording desynchronized. Skip this response
                # rather than lose the whole run's TPOT to it.
                unsynced += 1
                continue
            for (start, end), itl_value in zip(
                pairwise(offsets), output.itl, strict=True
            ):
                if end <= start:
                    continue
                pending.append(
                    (output, output.generated_text[start:end], itl_value)
                )
                if len(pending) < _TPOT_TOKENIZE_SLICE:
                    continue
                dropped_slice = _append_slice_tpots(
                    pending, tokenizer, failures
                )
                pending.clear()
                bar.update(1)
                if dropped_slice:
                    failed_slices += 1
                    dropped |= dropped_slice
                    # The refused slice held this response's newest chunk and
                    # cleared everything it had, so its remaining chunks would
                    # only record a tail.
                    break
        if pending:
            dropped_slice = _append_slice_tpots(pending, tokenizer, failures)
            bar.update(1)
            if dropped_slice:
                failed_slices += 1
                dropped |= dropped_slice

    if dropped or unsynced:
        detail = f" First failure: {failures[0]}." if failures else ""
        calls = "call" if failed_slices == 1 else "calls"
        logger.warning(
            "Per-chunk TPOT dropped for %d of %d responses: %d had chunk"
            " offsets out of step with their ITLs, and %d tokenizer %s"
            " failed.%s",
            len(dropped) + unsynced,
            len(outputs),
            unsynced,
            failed_slices,
            calls,
            detail,
        )


def _delta_text(delta: Mapping[str, Any], data: str) -> str:
    """Extracts all generated text carried by a chat streaming delta.

    ``reasoning`` and ``reasoning_content`` are NOT official OpenAI fields.
    Different model providers and serving frameworks may emit one or both to
    stream chain-of-thought tokens separately from ``content``. These fields
    may also be None in some chunks. We merge them here to preserve all
    streamed text.

    Tool-call fragments (function name and argument bytes) also count: a pure
    tool-call turn has no ``content`` by design, and without this a successful
    agentic response is misreported as "No text content captured".

    A field of the wrong type fails here, against the chunk that carried it,
    rather than surfacing as a surprising result in ``finish()``.
    """
    reasoning = _text_field(delta.get("reasoning"), "delta.reasoning", data)
    reasoning_content = _text_field(
        delta.get("reasoning_content"), "delta.reasoning_content", data
    )
    content = _text_field(delta.get("content"), "delta.content", data)
    tool_calls = delta.get("tool_calls")
    if content and not (reasoning or reasoning_content or tool_calls):
        # Fast path: a plain content chunk is nearly every chunk of a normal
        # generation.
        return content
    text = reasoning + reasoning_content + content
    for call in tool_calls or ():
        if not isinstance(call, Mapping):
            raise _malformed_chunk(
                "expected each tool call to be a JSON object", data
            )
        function = call.get("function")
        if isinstance(function, Mapping):
            text += _text_field(
                function.get("name"), "tool_calls[].function.name", data
            ) + _text_field(
                function.get("arguments"),
                "tool_calls[].function.arguments",
                data,
            )
    return text


class OpenAIStreamRecorder:
    """Measures one streamed OpenAI-style response.

    The measurement lands in a :class:`RequestFuncOutput`: TTFT at the first
    chunk carrying text, one inter-chunk latency per later text chunk, latency
    at the last event, and the server's usage counts.

    Per-chunk TPOT comes from :func:`compute_chunk_tpots` after the run, which
    re-tokenizes the recorded chunks.

    Args:
        prompt_len: The prompt length to record on the output.
        chat: Whether the stream carries chat completion chunks rather than
            completion chunks.
        tokenizer: The tokenizer for per-chunk TPOT, or ``None`` to skip it.
        is_cache_warmup: Accepts a completed, usage-bearing generation without
            visible text when its output is discarded for cache seeding.
        max_tokens: The request's output-token budget, or ``None`` if it set
            none. A budget of at most two tokens lets a measured response
            without visible text succeed; see :meth:`finish`.
        start: When the request was submitted, as a ``time.perf_counter()``
            value. A caller that spends time building the request passes the
            stamp it took before that work, so TTFT and latency cover it; the
            default reads the clock now.
    """

    def __init__(
        self,
        *,
        prompt_len: int,
        chat: bool,
        tokenizer: PreTrainedTokenizerBase | None = None,
        is_cache_warmup: bool = False,
        max_tokens: int | None = None,
        start: float | None = None,
    ) -> None:
        self.output = RequestFuncOutput()
        self.output.prompt_len = prompt_len
        self._chat = chat
        self._tokenizer = tokenizer
        self._generated_text: list[str] = []
        self._generated_chars = 0
        self._ttft = 0.0
        self._st = time.perf_counter() if start is None else start
        self.output.request_submit_time = self._st
        self._most_recent_timestamp = self._st
        self._has_content = False
        self._is_cache_warmup = is_cache_warmup
        self._max_tokens = max_tokens
        self._finish_reason: str | None = None
        self._finish_reason_time = 0.0
        self._stream_complete = False
        self._latency = 0.0
        self._done = False

    @property
    def done(self) -> bool:
        """Whether the request finished, was cancelled, or failed."""
        return self._done

    @property
    def stream_complete(self) -> bool:
        """Whether the stream reached ``[DONE]``, so reading can stop."""
        return self._stream_complete

    def on_event(
        self, data: str, *, arrived: float | None = None
    ) -> RequestFuncOutput | None:
        """Records one SSE event's data.

        Args:
            data: The event's data field.
            arrived: When the event arrived, as a ``time.perf_counter()``
                value; ``None`` reads the clock as the event is recorded.

        Returns:
            ``None`` to keep reading the stream, or the request's output when
            the event reported an error and the request is over.
        """
        # Stamped at arrival, before the parse, for latency, TTFT and ITL alike.
        timestamp = time.perf_counter() if arrived is None else arrived
        self._latency = timestamp - self._st
        if data == "[DONE]":
            self._stream_complete = True
            return None

        chunk = _decode_chunk(data)
        if not isinstance(chunk, Mapping):
            raise _malformed_chunk("expected a JSON object", data)

        error = chunk.get("error")
        if error is not None:
            message = (
                error.get("message") if isinstance(error, Mapping) else error
            )
            return self.fail(str(message or data))

        # Parse usage from any chunk that reports it.
        usage = chunk.get("usage")
        if usage:
            if not isinstance(usage, Mapping):
                raise _malformed_chunk(
                    "expected usage to be a JSON object", data
                )
            try:
                self.output.server_token_stats = _server_token_stats(usage)
            except TypeError as exc:
                raise _malformed_chunk(str(exc), data) from exc

        # A chunk with no choices just carries usage or a finish marker.
        choices = chunk.get("choices")
        if choices is None:
            return None
        if not isinstance(choices, list):
            raise _malformed_chunk("expected choices to be a JSON array", data)
        if not choices:
            return None
        choice = choices[0]
        if not isinstance(choice, Mapping):
            raise _malformed_chunk(
                "expected each choice to be a JSON object", data
            )
        finish_reason = choice.get("finish_reason")
        if finish_reason is not None:
            self._finish_reason = finish_reason
            self._finish_reason_time = timestamp
        if self._chat:
            delta = choice.get("delta")
            if not isinstance(delta, Mapping):
                raise _malformed_chunk(
                    "expected the choice to carry a delta object", data
                )
            if delta.get("tool_calls"):
                self.output.tool_call_returned = True
            text_content = _delta_text(delta, data)
        else:
            text_content = _required_text_field(
                choice.get("text"), "choice.text", data
            )

        # Only track timing for chunks with actual text
        if text_content:
            # A response only counts as content-bearing once it streams actual
            # text or tool-call fragments. Chunks that carry only a role or
            # finish_reason, or that put text in a delta field we don't model,
            # leave this False so the request is flagged rather than recorded
            # as a success with ttft=0 and no tokens.
            self._has_content = True
            # Kept together: the offset recorded below indexes this text, so the
            # two must never desync.
            self._generated_chars += len(text_content)
            self._generated_text.append(text_content)
            if self._tokenizer is not None:
                # Store end offsets, not chunk copies, to keep memory flat.
                self.output.pending_tpot_ends.append(self._generated_chars)
            # First token
            if self._ttft == 0.0:
                self._ttft = timestamp - self._st
                self.output.ttft = self._ttft

            # Decoding phase
            else:
                itl_value = timestamp - self._most_recent_timestamp
                self.output.itl.append(itl_value)

            self._most_recent_timestamp = timestamp
        return None

    def finish(self) -> RequestFuncOutput:
        """Marks a fully-received stream complete.

        Returns:
            The request's output.
        """
        self._done = True
        completion_tokens = self.output.server_token_stats.completion_tokens
        cache_warmup_completed = (
            self._is_cache_warmup
            and self._stream_complete
            and (completion_tokens or 0) > 0
        )
        # A one- or two-token budget can be spent entirely on tokens the
        # server's reasoning or tool parser consumes as delimiters (GLM's
        # ``</think>``), so a correct response may carry no visible text.
        # Accept that only when the server confirms it generated exactly the
        # tiny budget; every other measured response without text still fails.
        tiny_budget_exhausted = (
            self._stream_complete
            and self._finish_reason == "length"
            and self._max_tokens is not None
            and 0 < self._max_tokens <= 2
            and completion_tokens == self._max_tokens
        )
        if tiny_budget_exhausted and not self._has_content:
            self._ttft = self._finish_reason_time - self._st
            self.output.ttft = self._ttft
        if (
            not self._has_content
            and not cache_warmup_completed
            and not tiny_budget_exhausted
        ):
            self.output.error = (
                "No text content captured from the response"
                " (choices were present but"
                " delta.reasoning/reasoning_content/content/"
                "tool_calls were all empty). The model may stream"
                " text in a field this client does not parse."
            )
            self.output.success = False
        else:
            self.output.generated_text = "".join(self._generated_text)
            self.output.success = True
            self.output.latency = self._latency
        return self.output

    def fail(self, error: str) -> RequestFuncOutput:
        """Marks the request failed.

        Args:
            error: The failure's description.

        Returns:
            The request's output.
        """
        self._done = True
        self.output.error = error
        self.output.success = False
        return self.output

    def cancel(self) -> RequestFuncOutput:
        """Marks the request cancelled by the client.

        Resets the output, discarding all collected measurements.

        Returns:
            The request's output.
        """
        self._done = True
        self.output = RequestFuncOutput(
            cancelled=True, request_submit_time=time.perf_counter()
        )
        return self.output


def _body_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """*headers* with the JSON content type a ``json=`` body used to set.

    A pre-serialized body no longer lets aiohttp infer the content type, so the
    caller sets it here. The request's own headers win, and the dict passed in
    is left alone: the drivers cache their auth headers per run.
    """
    return {"Content-Type": "application/json", **headers}


async def _run_openai_stream_request(
    *,
    api_url: str,
    body: bytes,
    headers: Mapping[str, str],
    prompt_len: int,
    chat: bool,
    tokenizer: PreTrainedTokenizerBase | None = None,
    is_cache_warmup: bool = False,
    max_tokens: int | None = None,
    session: aiohttp.ClientSession,
    start: float | None = None,
) -> RequestFuncOutput:
    recorder = OpenAIStreamRecorder(
        prompt_len=prompt_len,
        chat=chat,
        tokenizer=tokenizer,
        is_cache_warmup=is_cache_warmup,
        max_tokens=max_tokens,
        start=start,
    )
    try:
        async with session.post(
            url=api_url, data=body, headers=_body_headers(headers)
        ) as response:
            if response.status != 200:
                return recorder.fail(response.reason or "")
            events = iter_events(response.content.iter_any())
            try:
                async for event in events:
                    result = recorder.on_event(event.data)
                    if result is not None:
                        return result
                    if recorder.stream_complete:
                        break
            finally:
                # Closed here rather than left to the garbage collector,
                # which reports the pending aclose() as a warning.
                await events.aclose()
            if recorder.stream_complete:
                # Drain to EOF so the socket closes cleanly and the connection
                # goes back to the pool this run shares.
                async for _ in response.content.iter_chunked(1 << 16):
                    pass
            return recorder.finish()
    except Exception:
        exc_info = sys.exc_info()
        return recorder.fail("".join(traceback.format_exception(*exc_info)))


class OpenAICompletionsRequestDriver(RequestDriver):
    """Request driver for OpenAI-compatible completions API."""

    async def request(
        self, request_func_input: BaseRequestFuncInput
    ) -> RequestFuncOutput:
        """Execute a request to the OpenAI-compatible completions API."""
        if not isinstance(request_func_input, RequestFuncInput):
            raise TypeError(
                "OpenAICompletionsRequestDriver requires RequestFuncInput."
            )
        api_url = request_func_input.api_url
        assert api_url.endswith(("completions", "profile")), (
            "OpenAI Completions API URL must end with 'completions' or 'profile'."
        )

        payload: dict[str, object] = {
            "model": request_func_input.model,
            "prompt": request_func_input.prompt,
            "best_of": 1,
            "stream": True,
            "stream_options": {"include_usage": True},
            "ignore_eos": request_func_input.ignore_eos,
            **self._sampling_fields(request_func_input.sampling),
        }

        if request_func_input.max_tokens is not None:
            payload["max_tokens"] = request_func_input.max_tokens
        payload = _build_final_payload(payload, self.extra_body)

        # The clock starts before the encode. On main aiohttp serialises the
        # payload inside ``session.post``, after the recorder has stamped the
        # start, so TTFT and latency include the encode; starting the clock
        # after one here would silently drop it from the measurement.
        start = time.perf_counter()
        body = msgspec.json.encode(payload)
        return await _run_openai_stream_request(
            api_url=api_url,
            body=body,
            headers=self._auth_headers(),
            prompt_len=request_func_input.prompt_len,
            chat=False,
            tokenizer=self.tokenizer,
            is_cache_warmup=request_func_input.is_cache_warmup,
            max_tokens=request_func_input.max_tokens,
            session=self.session,
            start=start,
        )


async def _run_atom_nonstream_chat_request(
    *,
    api_url: str,
    body: bytes,
    headers: Mapping[str, str],
    prompt_len: int,
    session: aiohttp.ClientSession,
    start: float | None = None,
) -> RequestFuncOutput:
    """ATOM workaround: non-streaming chat request using server-reported timing.

    ATOM returns the whole chat completion in one SSE chunk, so client-side
    stream timing is degenerate. Its non-streaming ``usage`` block reports
    ``ttft_s``/``tpot_s``/``latency_s`` instead; we use those and set
    ``generated_text`` so metrics derive TPOT as
    ``(latency - ttft)/(output_len - 1)`` (== ``usage.tpot_s``).

    TODO(ATOM): remove once ATOM streams chat/completions correctly.
    """
    output = RequestFuncOutput()
    output.prompt_len = prompt_len
    output.request_submit_time = time.perf_counter() if start is None else start

    try:
        async with session.post(
            url=api_url, data=body, headers=_body_headers(headers)
        ) as response:
            if response.status != 200:
                output.error = response.reason or ""
                output.success = False
                return output
            response_body = await response.json()
    except Exception:
        output.success = False
        output.error = "".join(traceback.format_exception(*sys.exc_info()))
        return output

    try:
        message = response_body["choices"][0].get("message") or {}
    except (KeyError, IndexError, TypeError):
        output.success = False
        output.error = f"Malformed chat completion response: {response_body!r}"
        return output

    # Merge reasoning/reasoning_content/content (ATOM puts <mm:think> in content).
    # Tool-call name and argument text counts too, as in the streaming path: a
    # pure tool-call response has no content by design.
    tool_calls = message.get("tool_calls") or []
    output.tool_call_returned = bool(tool_calls)
    generated_text = (
        (message.get("reasoning") or "")
        + (message.get("reasoning_content") or "")
        + (message.get("content") or "")
        + "".join(
            (fn.get("name") or "") + (fn.get("arguments") or "")
            for tc in tool_calls
            if isinstance(tc, dict)
            and isinstance(fn := tc.get("function"), dict)
        )
    )
    usage = response_body.get("usage") or {}
    ttft_s = usage.get("ttft_s")
    tpot_s = usage.get("tpot_s")
    latency_s = usage.get("latency_s")

    if ttft_s is None or latency_s is None:
        output.success = False
        output.error = (
            "ATOM server-reported timing requested but the response 'usage' is"
            " missing ttft_s/latency_s; this workaround only applies to an ATOM"
            " server that reports server-side timing on non-streaming"
            f" chat/completions. usage={usage!r}"
        )
        return output
    if not generated_text:
        output.success = False
        output.error = "No text content in chat completion response."
        return output

    output.generated_text = generated_text
    output.ttft = ttft_s
    # Metrics derive TPOT from (latency - ttft)/(output_len - 1) == usage.tpot_s.
    output.latency = latency_s
    prompt_details = usage.get("prompt_tokens_details") or {}
    completion_tokens = usage.get("completion_tokens")
    output.server_token_stats = ServerTokenStats(
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=completion_tokens,
        total_tokens=usage.get("total_tokens"),
        cached_tokens=prompt_details.get("cached_tokens", 0),
        cached_tokens_reported="cached_tokens" in prompt_details,
    )
    # No per-token latencies: synthesize a flat ITL/TPOT series from the mean
    # tpot_s so itl_ms/step_tpot_ms aren't NaN (headline TPOT still latency-based).
    if tpot_s is not None and completion_tokens and completion_tokens > 1:
        output.itl = [tpot_s] * (completion_tokens - 1)
        output.tpot = [tpot_s] * (completion_tokens - 1)
    output.success = True
    return output


class OpenAIChatCompletionsRequestDriver(RequestDriver):
    """Request driver for OpenAI-compatible chat completions API."""

    async def request(
        self, request_func_input: BaseRequestFuncInput
    ) -> RequestFuncOutput:
        """Execute a request to the OpenAI-compatible chat completions API."""
        if not isinstance(request_func_input, RequestFuncInput):
            raise TypeError(
                "OpenAIChatCompletionsRequestDriver requires RequestFuncInput."
            )
        api_url = request_func_input.api_url
        assert api_url.endswith("chat/completions"), (
            "OpenAI Chat Completions API URL must end with 'chat/completions'."
        )

        messages_json, images_attached = _chat_messages_json(
            request_func_input.prompt,
            request_func_input.images,
            request_func_input.wire_history,
        )

        payload: dict[str, object] = {
            "model": request_func_input.model,
            "messages": messages_json,
            "stream": True,
            "stream_options": {"include_usage": True},
            "ignore_eos": request_func_input.ignore_eos,
            **self._sampling_fields(request_func_input.sampling),
        }

        if request_func_input.max_tokens is not None:
            payload["max_tokens"] = request_func_input.max_tokens
        if request_func_input.response_format is not None:
            payload["response_format"] = request_func_input.response_format
        if request_func_input.tools:
            payload["tools"] = request_func_input.tools
        payload = _build_final_payload(payload, self.extra_body)

        headers = dict(self._auth_headers())
        if request_func_input.session_id:
            headers["X-Session-ID"] = request_func_input.session_id
        if request_func_input.cache_salt:
            headers["X-Cache-Salt"] = request_func_input.cache_salt

        # Stamped before the encode, which main leaves inside the measured
        # window; see the note in OpenAICompletionsRequestDriver.request.
        start = time.perf_counter()
        if self.backend == "atom":
            # ATOM doesn't per-token-stream chat/completions: send non-streaming
            # and read timing from `usage`. TODO(ATOM): remove once it streams.
            nonstream_payload = dict(payload)
            nonstream_payload["stream"] = False
            nonstream_payload.pop("stream_options", None)
            output = await _run_atom_nonstream_chat_request(
                api_url=api_url,
                body=msgspec.json.encode(nonstream_payload),
                headers=headers,
                prompt_len=request_func_input.prompt_len,
                session=self.session,
                start=start,
            )
        else:
            output = await _run_openai_stream_request(
                api_url=api_url,
                body=msgspec.json.encode(payload),
                headers=headers,
                prompt_len=request_func_input.prompt_len,
                chat=True,
                tokenizer=self.tokenizer,
                is_cache_warmup=request_func_input.is_cache_warmup,
                max_tokens=request_func_input.max_tokens,
                session=self.session,
                start=start,
            )
        output.tools_offered = "tools" in payload
        output.carries_image = (
            images_attached or request_func_input.prompt_carries_image
        )
        return output


_GENERATED_MEDIA_TYPES = frozenset({"output_image", "output_video"})


def _count_generated_media(data: dict[str, Any]) -> int:
    output = data.get("output")
    if not isinstance(output, list):
        logger.warning(
            f"OpenResponses response has unexpected 'output' type: "
            f"{type(output).__name__}"
        )
        return 0

    count = 0

    for message_idx, message in enumerate(output):
        if not isinstance(message, dict):
            logger.warning(
                f"Skipping output[{message_idx}]: expected dict, got {type(message)}."
            )
            continue
        content = message.get("content")
        if not isinstance(content, list):
            logger.warning(
                f"Skipping output[{message_idx}].content: expected list, got {type(content)}."
            )
            continue
        for item_idx, item in enumerate(content):
            if not isinstance(item, dict):
                logger.warning(
                    f"Skipping output[{message_idx}].content[{item_idx}]: expected dict, got {type(item)}."
                )
                continue
            if item.get("type") in _GENERATED_MEDIA_TYPES:
                count += 1

    return count


def _build_pixel_generation_payload(
    request_func_input: PixelGenerationRequestFuncInput,
) -> dict[str, Any]:
    input_payload: str | list[dict[str, Any]]
    if request_func_input.input_image_paths:
        content: list[dict[str, Any]] = []
        for image_path in request_func_input.input_image_paths:
            content.append(
                _encode_openresponses_image_from_file_path(image_path)
            )
        content.append(
            {"type": "input_text", "text": request_func_input.prompt}
        )
        input_payload = [{"role": "user", "content": content}]
    else:
        input_payload = request_func_input.prompt

    payload: dict[str, Any] = {
        "model": request_func_input.model,
        "input": input_payload,
    }

    if request_func_input.image_options is None:
        return payload

    options_payload: dict[str, Any] = {}
    image_options = request_func_input.image_options
    if image_options.width is not None:
        options_payload["width"] = image_options.width
    if image_options.height is not None:
        options_payload["height"] = image_options.height
    if image_options.steps is not None:
        options_payload["steps"] = image_options.steps
    if image_options.guidance_scale is not None:
        options_payload["guidance_scale"] = image_options.guidance_scale
    if image_options.negative_prompt is not None:
        options_payload["negative_prompt"] = image_options.negative_prompt
    # num_frames is video-only; presence routes the payload to
    # provider_options.video instead of provider_options.image.
    is_video = image_options.num_frames is not None
    if is_video:
        options_payload["num_frames"] = image_options.num_frames

    if options_payload:
        modality_key = "video" if is_video else "image"
        payload["provider_options"] = {modality_key: options_payload}

    if image_options.seed is not None:
        payload["seed"] = image_options.seed

    return payload


class OpenResponsesRequestDriver(RequestDriver):
    """Request driver for OpenResponses API."""

    async def request(
        self, request_func_input: BaseRequestFuncInput
    ) -> PixelGenerationRequestFuncOutput:
        """Execute a request to the OpenResponses API."""
        if not isinstance(request_func_input, PixelGenerationRequestFuncInput):
            raise TypeError(
                "OpenResponsesRequestDriver requires PixelGenerationRequestFuncInput."
            )
        api_url = request_func_input.api_url
        assert api_url.endswith("responses"), (
            "OpenResponses API URL must end with 'responses'."
        )

        payload = _build_pixel_generation_payload(request_func_input)

        headers = {
            "Content-Type": "application/json",
            **self._auth_headers(),
        }

        output = PixelGenerationRequestFuncOutput()
        start = time.perf_counter()
        output.request_submit_time = start
        try:
            async with self.session.post(
                url=api_url, json=payload, headers=headers
            ) as response:
                output.latency = time.perf_counter() - start
                if response.status != 200:
                    body = await response.text()
                    output.error = (
                        f"HTTP {response.status}: {body}"
                        if body
                        else (response.reason or "")
                    )
                    output.success = False
                    return output

                body = await response.json()
                output.num_generated_outputs = _count_generated_media(body)
                if output.num_generated_outputs <= 0:
                    output.error = (
                        "No output_image or output_video content found in"
                        " OpenResponses response body."
                    )
                    output.success = False
                    return output

                output.success = True
                return output
        except Exception:
            output.latency = time.perf_counter() - start
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))
            return output


def _build_sglang_pixel_generation_payload(
    request_func_input: PixelGenerationRequestFuncInput,
) -> dict[str, Any]:
    """Build payload for sglang's /v1/images/generations endpoint."""
    payload: dict[str, Any] = {
        "model": request_func_input.model,
        "prompt": request_func_input.prompt,
        "n": 1,
        "response_format": "b64_json",
    }

    if request_func_input.image_options is not None:
        opts = request_func_input.image_options
        if opts.width is not None and opts.height is not None:
            payload["size"] = f"{opts.width}x{opts.height}"
        if opts.steps is not None:
            payload["num_inference_steps"] = opts.steps
        if opts.guidance_scale is not None:
            payload["guidance_scale"] = opts.guidance_scale
        if opts.seed is not None:
            payload["seed"] = opts.seed
        # negative_prompt is not supported by sglang's images API.

    return payload


class SglangPixelGenerationRequestDriver(RequestDriver):
    """Request driver for sglang's /v1/images/generations endpoint."""

    async def request(
        self, request_func_input: BaseRequestFuncInput
    ) -> PixelGenerationRequestFuncOutput:
        """Sends a pixel generation request to sglang."""
        if not isinstance(request_func_input, PixelGenerationRequestFuncInput):
            raise TypeError(
                "SglangPixelGenerationRequestDriver requires"
                " PixelGenerationRequestFuncInput."
            )
        api_url = request_func_input.api_url
        if not api_url.endswith("images/generations"):
            raise ValueError(
                "Sglang pixel generation URL must end with"
                " 'images/generations'."
            )

        payload = _build_sglang_pixel_generation_payload(request_func_input)

        headers = {
            "Content-Type": "application/json",
            **self._auth_headers(),
        }

        output = PixelGenerationRequestFuncOutput()
        start = time.perf_counter()
        output.request_submit_time = start
        try:
            async with self.session.post(
                url=api_url, json=payload, headers=headers
            ) as response:
                output.latency = time.perf_counter() - start
                if response.status != 200:
                    body = await response.text()
                    output.error = (
                        f"HTTP {response.status}: {body}"
                        if body
                        else (response.reason or "")
                    )
                    output.success = False
                    return output

                body = await response.json()
                # sglang returns {"data": [{"b64_json": "..."}, ...]}
                data = body.get("data", [])
                output.num_generated_outputs = len(data)
                if output.num_generated_outputs <= 0:
                    output.error = "No images found in sglang response body."
                    output.success = False
                    return output

                output.success = True
                return output
        except Exception:
            output.latency = time.perf_counter() - start
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))
            return output


def _add_input_reference(
    form: aiohttp.FormData, input_image_paths: list[str] | None
) -> None:
    """Attach the i2v conditioning image to a multipart form as ``input_reference``.

    vllm-omni (``/v1/videos[/sync]``) and sglang (``/v1/videos``) both take the
    image-to-video conditioning image as a multipart ``input_reference`` file.
    Only the first image is sent; both server APIs accept a single reference.
    """
    if not input_image_paths:
        return
    image_path = input_image_paths[0]
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    form.add_field(
        "input_reference",
        image_bytes,
        filename=os.path.basename(image_path),
        content_type="application/octet-stream",
    )


class SglangVideoPayload(TypedDict):
    """The JSON body of a sglang video generation request."""

    model: str
    prompt: str
    width: NotRequired[int]
    height: NotRequired[int]
    num_inference_steps: NotRequired[int]
    guidance_scale: NotRequired[float]
    seed: NotRequired[int]
    negative_prompt: NotRequired[str]
    num_frames: NotRequired[int]


def _build_sglang_video_payload(
    request_func_input: PixelGenerationRequestFuncInput,
) -> SglangVideoPayload | Mapping[str, object]:
    """Build JSON payload for sglang's POST /v1/videos endpoint."""
    options = request_func_input.image_options
    schema = get_video_request_schema(
        options.request_schema if options else None
    )
    if schema is not None:
        return schema.build_payload(request_func_input)

    payload = SglangVideoPayload(
        model=request_func_input.model,
        prompt=request_func_input.prompt,
    )

    if request_func_input.image_options is not None:
        opts = request_func_input.image_options
        if opts.width is not None:
            payload["width"] = opts.width
        if opts.height is not None:
            payload["height"] = opts.height
        if opts.steps is not None:
            payload["num_inference_steps"] = opts.steps
        if opts.guidance_scale is not None:
            payload["guidance_scale"] = opts.guidance_scale
        if opts.seed is not None:
            payload["seed"] = opts.seed
        if opts.negative_prompt is not None:
            payload["negative_prompt"] = opts.negative_prompt
        if opts.num_frames is not None:
            payload["num_frames"] = opts.num_frames

    return payload


def _build_sglang_video_form(
    request_func_input: PixelGenerationRequestFuncInput,
) -> aiohttp.FormData:
    """Build the multipart form for an image-to-video sglang ``/v1/videos`` request.

    Mirrors sglang's reference ``multimodal_gen`` bench_serving: ``size=WxH``
    and ``num_frames`` are top-level form fields, the remaining sampling knobs
    are JSON-encoded under ``extra_body``, and the conditioning image is the
    ``input_reference`` file.
    """
    form = aiohttp.FormData()
    form.add_field("model", request_func_input.model)
    form.add_field("prompt", request_func_input.prompt)

    extra_body: dict[str, Any] = {}
    opts = request_func_input.image_options
    if opts is not None:
        if opts.width is not None and opts.height is not None:
            form.add_field("size", f"{opts.width}x{opts.height}")
        if opts.num_frames is not None:
            form.add_field("num_frames", str(opts.num_frames))
        if opts.steps is not None:
            extra_body["num_inference_steps"] = opts.steps
        if opts.guidance_scale is not None:
            extra_body["guidance_scale"] = opts.guidance_scale
        if opts.seed is not None:
            extra_body["seed"] = opts.seed
        if opts.negative_prompt is not None:
            extra_body["negative_prompt"] = opts.negative_prompt
    if extra_body:
        form.add_field("extra_body", json.dumps(extra_body))

    _add_input_reference(form, request_func_input.input_image_paths)
    return form


_SGLANG_VIDEO_POLL_INTERVAL_S = 1.0


async def _fetch_sglang_video_content(
    session: aiohttp.ClientSession,
    poll_url: str,
    headers: dict[str, str],
    output: PixelGenerationRequestFuncOutput,
) -> bool:
    """Downloads the mp4 of a completed sglang video job.

    Returns whether the download succeeded; on failure the reason is recorded on
    ``output.error``.
    """
    async with session.get(url=f"{poll_url}/content", headers=headers) as resp:
        if resp.status != 200:
            body = await resp.text()
            output.error = (
                f"Content HTTP {resp.status}: {body}"
                if body
                else (resp.reason or "")
            )
            return False
        if not await resp.read():
            output.error = "Empty mp4 in sglang video content response."
            return False
    return True


class SglangVideoRequestDriver(RequestDriver):
    """Request driver for sglang's async /v1/videos endpoint.

    POST /v1/videos queues a job, then poll GET /v1/videos/{id} until done.
    Ref: https://github.com/sgl-project/sglang/blob/v0.5.10.post1/python/sglang/multimodal_gen/benchmarks/bench_serving.py#L224
    """

    async def request(
        self, request_func_input: BaseRequestFuncInput
    ) -> PixelGenerationRequestFuncOutput:
        """Sends a video generation request to sglang."""
        if not isinstance(request_func_input, PixelGenerationRequestFuncInput):
            raise TypeError(
                "SglangVideoRequestDriver requires"
                " PixelGenerationRequestFuncInput."
            )
        api_url = request_func_input.api_url
        if not api_url.rstrip("/").endswith("/videos"):
            raise ValueError("Sglang video URL must end with '/videos'.")
        base_url = api_url.rstrip("/")

        options = request_func_input.image_options
        schema = get_video_request_schema(
            options.request_schema if options else None
        )
        # image-to-video uploads the conditioning image, which requires
        # multipart/form-data; text-to-video stays JSON. Let aiohttp set the
        # multipart Content-Type (with boundary) for the form path.
        headers = dict(self._auth_headers())
        if request_func_input.input_image_paths and schema is None:
            post_kwargs: dict[str, Any] = {
                "data": _build_sglang_video_form(request_func_input)
            }
        else:
            headers["Content-Type"] = "application/json"
            post_kwargs = {
                "json": _build_sglang_video_payload(request_func_input)
            }

        output = PixelGenerationRequestFuncOutput()
        start = time.perf_counter()
        output.request_submit_time = start

        try:
            async with self.session.post(
                url=base_url, headers=headers, **post_kwargs
            ) as response:
                if response.status != 200:
                    body = await response.text()
                    output.latency = time.perf_counter() - start
                    output.error = (
                        f"HTTP {response.status}: {body}"
                        if body
                        else (response.reason or "")
                    )
                    output.success = False
                    return output

                body = await response.json()
                job_id = body.get("id")
                if not job_id:
                    output.latency = time.perf_counter() - start
                    output.error = "No job id in sglang video POST response."
                    output.success = False
                    return output

            poll_url = f"{base_url}/{job_id}"
            while True:
                async with self.session.get(
                    url=poll_url, headers=headers
                ) as poll_response:
                    if poll_response.status != 200:
                        body = await poll_response.text()
                        output.latency = time.perf_counter() - start
                        output.error = (
                            f"Poll HTTP {poll_response.status}: {body}"
                            if body
                            else (poll_response.reason or "")
                        )
                        output.success = False
                        return output

                    poll_body = await poll_response.json()
                    status = poll_body.get("status", "")

                    if status == "completed":
                        if (
                            schema is not None
                            and not await _fetch_sglang_video_content(
                                self.session, poll_url, headers, output
                            )
                        ):
                            output.latency = time.perf_counter() - start
                            output.success = False
                            return output
                        output.latency = time.perf_counter() - start
                        output.num_generated_outputs = 1
                        output.success = True

                        inference_time = poll_body.get("inference_time_s")
                        if inference_time:
                            logger.debug(
                                "sglang video: inference_time=%s s",
                                inference_time,
                            )
                        return output

                    if status in ("failed", "error"):
                        output.latency = time.perf_counter() - start
                        error_info = poll_body.get("error", {})
                        output.error = (
                            error_info.get("message", "")
                            if isinstance(error_info, dict)
                            else str(error_info)
                        )
                        output.success = False
                        return output

                    await asyncio.sleep(_SGLANG_VIDEO_POLL_INTERVAL_S)

        except Exception:
            output.latency = time.perf_counter() - start
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))
            return output


def _build_vllm_omni_pixel_generation_payload(
    request_func_input: PixelGenerationRequestFuncInput,
) -> dict[str, Any]:
    """Build payload for vllm-omni's /v1/chat/completions endpoint."""
    extra_body: dict[str, Any] = {}

    if request_func_input.image_options is not None:
        opts = request_func_input.image_options
        if opts.height is not None:
            extra_body["height"] = opts.height
        if opts.width is not None:
            extra_body["width"] = opts.width
        if opts.steps is not None:
            extra_body["num_inference_steps"] = opts.steps
        if opts.guidance_scale is not None:
            extra_body["guidance_scale"] = opts.guidance_scale
        if opts.seed is not None:
            extra_body["seed"] = opts.seed
        # negative_prompt is not supported by vllm-omni's chat API.

    payload: dict[str, Any] = {
        "model": request_func_input.model,
        "messages": [{"role": "user", "content": request_func_input.prompt}],
    }
    if extra_body:
        payload["extra_body"] = extra_body

    return payload


class VllmOmniPixelGenerationRequestDriver(RequestDriver):
    """Request driver for vllm-omni's chat completions endpoint.

    Targets ``/v1/chat/completions`` for diffusion image generation.
    """

    async def request(
        self, request_func_input: BaseRequestFuncInput
    ) -> PixelGenerationRequestFuncOutput:
        """Sends a pixel generation request to vllm-omni."""
        if not isinstance(request_func_input, PixelGenerationRequestFuncInput):
            raise TypeError(
                "VllmOmniPixelGenerationRequestDriver requires"
                " PixelGenerationRequestFuncInput."
            )
        api_url = request_func_input.api_url
        if not api_url.endswith("chat/completions"):
            raise ValueError(
                "vllm-omni pixel generation URL must end with"
                " 'chat/completions'."
            )

        payload = _build_vllm_omni_pixel_generation_payload(request_func_input)

        headers = {
            "Content-Type": "application/json",
            **self._auth_headers(),
        }

        output = PixelGenerationRequestFuncOutput()
        start = time.perf_counter()
        output.request_submit_time = start
        try:
            async with self.session.post(
                url=api_url, json=payload, headers=headers
            ) as response:
                output.latency = time.perf_counter() - start
                if response.status != 200:
                    body = await response.text()
                    output.error = (
                        f"HTTP {response.status}: {body}"
                        if body
                        else (response.reason or "")
                    )
                    output.success = False
                    return output

                body = await response.json()
                # vllm-omni returns chat completions format with image
                # data in choices[].message.content[].image_url.url
                choices = body.get("choices", [])
                count = 0
                for choice in choices:
                    message = choice.get("message", {})
                    content = message.get("content", [])
                    if isinstance(content, list):
                        count += sum(
                            1
                            for item in content
                            if isinstance(item, dict)
                            and item.get("type") == "image_url"
                        )

                output.num_generated_outputs = count
                if output.num_generated_outputs <= 0:
                    output.error = "No images found in vllm-omni response body."
                    output.success = False
                    return output

                output.success = True
                return output
        except Exception:
            output.latency = time.perf_counter() - start
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))
            return output


def _build_vllm_omni_video_payload(
    request_func_input: PixelGenerationRequestFuncInput,
) -> dict[str, str]:
    """Build form payload for vllm-omni's /v1/videos/sync endpoint."""
    payload: dict[str, str] = {
        "prompt": request_func_input.prompt,
        "model": request_func_input.model,
    }

    if request_func_input.image_options is not None:
        opts = request_func_input.image_options
        if opts.width is not None:
            payload["width"] = str(opts.width)
        if opts.height is not None:
            payload["height"] = str(opts.height)
        if opts.steps is not None:
            payload["num_inference_steps"] = str(opts.steps)
        if opts.guidance_scale is not None:
            payload["guidance_scale"] = str(opts.guidance_scale)
        if opts.seed is not None:
            payload["seed"] = str(opts.seed)
        if opts.negative_prompt is not None:
            payload["negative_prompt"] = opts.negative_prompt
        if opts.num_frames is not None:
            payload["num_frames"] = str(opts.num_frames)

    return payload


class VllmOmniVideoRequestDriver(RequestDriver):
    """Request driver for vllm-omni's videos endpoint.

    Targets ``/v1/videos/sync`` for diffusion video generation.
    """

    async def request(
        self, request_func_input: BaseRequestFuncInput
    ) -> PixelGenerationRequestFuncOutput:
        """Sends a video generation request to vllm-omni."""
        if not isinstance(request_func_input, PixelGenerationRequestFuncInput):
            raise TypeError(
                "VllmOmniVideoRequestDriver requires"
                " PixelGenerationRequestFuncInput."
            )
        api_url = request_func_input.api_url
        if not api_url.endswith("videos/sync"):
            raise ValueError(
                "vllm-omni video generation URL must end with 'videos/sync'."
            )

        # For image-to-video the conditioning image rides along as the
        # `input_reference` file, which requires multipart/form-data. The
        # text-to-video path keeps its existing form-encoded dict POST.
        payload = _build_vllm_omni_video_payload(request_func_input)
        if request_func_input.input_image_paths:
            form = aiohttp.FormData()
            for field_name, value in payload.items():
                form.add_field(field_name, value)
            _add_input_reference(form, request_func_input.input_image_paths)
            post_data: Any = form
        else:
            post_data = payload

        # A copy per request: the cached dict is shared by every request.
        headers = dict(self._auth_headers())

        output = PixelGenerationRequestFuncOutput()

        start = time.perf_counter()
        output.request_submit_time = start
        try:
            async with self.session.post(
                url=api_url, data=post_data, headers=headers
            ) as response:
                output.latency = time.perf_counter() - start
                if response.status != 200:
                    body = await response.text()
                    output.error = (
                        f"HTTP {response.status}: {body}"
                        if body
                        else (response.reason or "")
                    )
                    output.success = False
                    return output

                video_bytes = await response.read()
                if not video_bytes:
                    output.error = "Empty response body from /v1/videos/sync."
                    output.success = False
                    return output

                output.num_generated_outputs = 1
                output.success = True

                inference_time = response.headers.get("X-Inference-Time-S")
                if inference_time:
                    logger.debug(
                        "vllm-omni video: inference_time=%s s",
                        inference_time,
                    )

                return output
        except Exception:
            output.latency = time.perf_counter() - start
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))
            return output


class RequestCounter:
    """Thread-safe counter for limiting the number of requests in benchmarks.

    This class provides a simple mechanism to track and limit the total number
    of requests sent across multiple concurrent threads. It uses a threading.Lock
    to ensure thread-safe access to the counter.

    """

    def __init__(
        self,
        max_requests: int,
        total_sent_requests: int = 0,
    ) -> None:
        """Initialize the request counter.

        Args:
            max_requests: Maximum number of requests allowed
            total_sent_requests: Initial count of sent requests (default: 0)
        """
        self.max_requests = max_requests
        self.req_counter_lock = threading.Lock()
        self.total_sent_requests = total_sent_requests

    def advance_until_max(self) -> bool:
        """Atomically check and increment the request counter.

        This method performs a thread-safe check-and-increment operation.
        If the current count is below max_requests, it increments the counter
        and returns True. If the limit has been reached, it returns False.

        Returns:
            True if the request can proceed (counter was incremented),
            False if max_requests has been reached.
        """
        with self.req_counter_lock:
            if self.total_sent_requests >= self.max_requests:
                return False

            self.total_sent_requests += 1
            if self.total_sent_requests == self.max_requests:
                logger.info(
                    f"Request cap reached (--num-prompts={self.max_requests}):"
                    " no new requests will start; waiting for queued and"
                    " in-flight requests to complete."
                )
            return True


async def async_request_lora_load(
    api_url: str, lora_name: str, lora_path: str
) -> tuple[bool, float]:
    """Load a LoRA adapter via the API.

    Returns:
        Tuple of (success, load_time_ms)
    """
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {"lora_name": lora_name, "lora_path": lora_path}
        headers = {"Content-Type": "application/json"}
        logger.debug(f"Loading LoRA '{lora_name}' from path: {lora_path}")

        start_time = time.perf_counter()
        try:
            async with session.post(
                url=f"{api_url}/v1/load_lora_adapter",
                json=payload,
                headers=headers,
            ) as response:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                if response.status == 200:
                    logger.debug(
                        f"Successfully loaded LoRA '{lora_name}' in"
                        f" {elapsed_ms:.2f}ms"
                    )
                    return True, elapsed_ms
                else:
                    error_text = await response.text()
                    logger.error(
                        f"Failed to load LoRA '{lora_name}': {error_text}"
                    )
                    return False, elapsed_ms
        except Exception:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.exception(f"Exception loading LoRA '{lora_name}'")
            return False, elapsed_ms


async def async_request_lora_unload(
    api_url: str, lora_name: str
) -> tuple[bool, float]:
    """Unload a LoRA adapter via the API.

    Returns:
        Tuple of (success, unload_time_ms)
    """
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {"lora_name": lora_name}
        headers = {"Content-Type": "application/json"}

        start_time = time.perf_counter()
        try:
            async with session.post(
                url=f"{api_url}/v1/unload_lora_adapter",
                json=payload,
                headers=headers,
            ) as response:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                if response.status == 200:
                    logger.debug(
                        f"Successfully unloaded LoRA '{lora_name}' in"
                        f" {elapsed_ms:.2f}ms"
                    )
                    return True, elapsed_ms
                else:
                    error_text = await response.text()
                    logger.error(
                        f"Failed to unload LoRA '{lora_name}': {error_text}"
                    )
                    return False, elapsed_ms
        except Exception:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.exception(f"Exception unloading LoRA '{lora_name}'")
            return False, elapsed_ms


def get_request_driver_class(
    api_url: str,
    task: BenchmarkTask = "text-generation",
) -> type[RequestDriver]:
    """Return the request driver based on endpoint and optional task.

    For pixel generation, driver selection is based on URL suffix because
    each backend uses a fundamentally different API format. The mapping is:
      /v1/responses          -> OpenResponsesRequestDriver (modular)
      /v1/images/generations -> SglangPixelGenerationRequestDriver
      /v1/videos/sync        -> VllmOmniVideoRequestDriver
      /v1/videos             -> SglangVideoRequestDriver
      /v1/chat/completions   -> VllmOmniPixelGenerationRequestDriver
    The correct endpoint is typically auto-selected by PIXEL_GEN_DEFAULT_ENDPOINT
    (or VIDEO_GEN_DEFAULT_ENDPOINT for text-to-video) in benchmark_serving.py
    based on the backend, task and any registered video request schema.
    """
    if task in PIXEL_GENERATION_TASKS:
        if api_url.endswith("responses"):
            return OpenResponsesRequestDriver
        if api_url.endswith("images/generations"):
            return SglangPixelGenerationRequestDriver
        if api_url.endswith("videos/sync"):
            return VllmOmniVideoRequestDriver
        if api_url.endswith("/videos"):
            return SglangVideoRequestDriver
        if api_url.endswith("chat/completions"):
            return VllmOmniPixelGenerationRequestDriver
        raise ValueError(
            "Unsupported API URL for pixel-generation driver selection: "
            f"'{api_url}'. Expected /v1/responses, /v1/images/generations,"
            " /v1/videos/sync, /v1/videos, or /v1/chat/completions."
        )

    # for text generation task
    if api_url.endswith("chat/completions"):
        return OpenAIChatCompletionsRequestDriver
    if api_url.endswith(("completions", "profile")):
        return OpenAICompletionsRequestDriver
    if api_url.endswith("generate_stream"):
        return TRTLLMRequestDriver
    raise ValueError(
        "Unsupported API URL for request driver selection: "
        f"'{api_url}'. Expected an OpenAI completions/chat endpoint or "
        "TensorRT-LLM generate_stream endpoint."
    )
