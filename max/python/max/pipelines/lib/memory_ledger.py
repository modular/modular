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
"""Sizes a pipeline's KV cache from what its warmups allocate.

A pipeline loads its weights and compiles its graphs once, against a
:class:`~max.pipelines.kv_cache.NullKVCache` that holds only the null page.
:func:`size_kv_cache` then runs the forwards whose workspaces are largest --
the widest prefill from an empty cache and at the longest context, the
largest decode batch at the longest context, and for a vision model a
prefill chunk filled with the largest images, and for a video model the
first chunk of the largest clip, whose steps run the vision encoder -- and
reads each device's memory manager:

* resident: what the device holds before the warmups (weights, compiled
  graphs, persistent buffers, the null cache);
* eager: the largest peak any one warmup forward adds on top;
* capture: with graph capture on, the decode warmup's peak again, for the
  pool the captured graphs keep for good beside the eager forwards.

The cache gets, on the device with the least left, ``total *
device_memory_utilization - resident - eager - capture - margin``; ``max_length``
is clamped to what that cache holds, and an unset ``max_batch_size`` is
halved while the warmups run out of memory or leave no room for one prefill
chunk. The pipeline then swaps the null cache for the real one. Graph
capture runs after, against the real cache's addresses, and
:func:`check_capture` holds it to its share: the graphs a model captures
share one memory pool and replay one at a time, so they keep about what
the largest decode did.

Every boot measures; nothing is stored between boots.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import dataclasses
import gc
import logging
import os
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from max.driver import Device, is_virtual_device_mode
from max.support.human_readable_formatter import to_human_readable_bytes

if TYPE_CHECKING:
    from .memory_plan import MemoryPlan

__all__ = [
    "DISABLE_ENV",
    "MARGIN_ENV",
    "DeviceLedger",
    "Ledger",
    "cache_budget",
    "check_capture",
    "measures",
    "measuring_disabled",
    "measuring_off",
    "run_warmups",
    "size_kv_cache",
    "unmeasured_budget",
]

logger = logging.getLogger("max.pipelines")

GIB = 1 << 30

DEFAULT_MARGIN_BYTES = GIB
"""Held back on every device on top of the measurement."""

UNMEASURED_TRANSIENT_BYTES = 6 * GIB
"""Held back per device when nothing can be measured (no memory manager, a
host-only pipeline, or measuring turned off)."""

MARGIN_ENV = "MAX_MEMORY_LEDGER_MARGIN_BYTES"
DISABLE_ENV = "MAX_MEMORY_MEASURE_DISABLE"


def margin_bytes() -> int:
    """The per-device margin, overridable with ``MAX_MEMORY_LEDGER_MARGIN_BYTES``."""
    value = os.environ.get(MARGIN_ENV)
    return int(value) if value else DEFAULT_MARGIN_BYTES


def measuring_disabled() -> bool:
    """Whether ``MAX_MEMORY_MEASURE_DISABLE=1`` turns measuring off."""
    return os.environ.get(DISABLE_ENV) == "1"


@contextlib.contextmanager
def measuring_off() -> Iterator[None]:
    """Turns measuring off for a block.

    A tool that builds a pipeline only to record its graphs sizes the cache
    with the fixed reserve instead of running warmups.
    """
    previous = os.environ.get(DISABLE_ENV)
    os.environ[DISABLE_ENV] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(DISABLE_ENV, None)
        else:
            os.environ[DISABLE_ENV] = previous


def measures(devices: Sequence[Device]) -> bool:
    """Whether the planner leaves the budget to a measurement on ``devices``.

    Read before any device memory is touched, so it cannot ask the memory
    manager; :func:`measurable` asks once the pipeline holds its weights.
    Only GPUs measure. An accelerator that compiles its programs ahead
    allocates their memory as it loads, so the planner sizes its cache from
    what is free before loading, as :func:`unmeasured_budget` describes.
    """
    return (
        not measuring_disabled()
        and not is_virtual_device_mode()
        and bool(devices)
        and all(_is_gpu(d) for d in devices)
    )


def _is_gpu(device: Device) -> bool:
    """Whether ``device`` is a GPU, which allocates its transients per step."""
    return not device.is_host and device.api in ("cuda", "hip")


# ---------------------------------------------------------------------------
# Device statistics


@dataclass
class DeviceLedger:
    """What one device holds, measured around the warmups."""

    total_bytes: int
    resident_bytes: int
    eager_bytes: int = 0
    """The largest peak any one warmup forward added."""
    capture_bytes: int = 0
    """What captured graphs keep for good: the decode warmup's peak when
    graph capture is on, since every captured graph shares one pool."""
    driver_fallbacks: int = 0

    @property
    def transient_bytes(self) -> int:
        """Eager workspaces on top of what captured graphs keep."""
        return self.eager_bytes + self.capture_bytes

    def usable_bytes(self, utilization: float) -> int:
        """Memory this device may give the KV cache under ``utilization``."""
        return int(self.total_bytes * utilization) - self.resident_bytes

    def cache_bytes(self, utilization: float, margin: int) -> int:
        """What this device leaves for the cache once every claim is paid."""
        return self.usable_bytes(utilization) - self.transient_bytes - margin


def _used_bytes(device: Device) -> int:
    """Device memory in use, from the driver: total minus free."""
    stats = device.stats
    free = stats.get("free_memory")
    total = int(stats["total_memory"])
    return total - int(free) if free is not None else 0


def _live_bytes(device: Device) -> int:
    return int(device.memory_manager_stats["live_bytes"])


def measurable(devices: Sequence[Device]) -> bool:
    """Whether every device is an accelerator whose memory manager reports."""
    if not devices or any(d.is_host for d in devices):
        return False
    try:
        return all(_live_bytes(d) > 0 for d in devices)
    except Exception:
        return False


def cache_budget(ledgers: Sequence[DeviceLedger], utilization: float) -> int:
    """KV-cache bytes across all devices the ledgers leave."""
    margin = margin_bytes()
    per_device = min(m.cache_bytes(utilization, margin) for m in ledgers)
    return max(0, per_device) * len(ledgers)


def log_ledger(ledgers: Sequence[DeviceLedger], utilization: float) -> None:
    """Logs each device's ledger: usable, claims, and the cache it leaves."""
    margin = margin_bytes()
    lines = [
        f"  {'device':>6} {'resident':>12} {'usable':>12} {'eager':>12} "
        f"{'capture':>12} {'margin':>10} {'cache':>12}"
    ]
    for i, m in enumerate(ledgers):
        lines.append(
            f"  {i:>6} {to_human_readable_bytes(m.resident_bytes):>12} "
            f"{to_human_readable_bytes(m.usable_bytes(utilization)):>12} "
            f"{to_human_readable_bytes(m.eager_bytes):>12} "
            f"{to_human_readable_bytes(m.capture_bytes):>12} "
            f"{to_human_readable_bytes(margin):>10} "
            f"{to_human_readable_bytes(max(0, m.cache_bytes(utilization, margin))):>12}"
        )
    logger.info(
        "memory ledger (device_memory_utilization=%s):\n%s",
        utilization,
        "\n".join(lines),
    )


def _unmeasured_share(device: Device, utilization: float) -> int:
    """What an unmeasured device may give the cache.

    A GPU allocates its transients per step: it gets its share of
    ``total * utilization`` less what it holds and a fixed transient reserve.
    An accelerator that compiles its programs ahead allocated their memory at
    load, outside any budget, so it gets ``utilization`` of what is free
    (before weights load, exactly what the per-model estimators gave it).
    One that cannot report what is free (a plugin HAL's ``None``) gets
    ``utilization`` of its total, the most that could be free.
    """
    total = int(device.stats["total_memory"])
    if _is_gpu(device):
        return (
            int(total * utilization)
            - _used_bytes(device)
            - UNMEASURED_TRANSIENT_BYTES
        )
    free = device.stats.get("free_memory")
    return int((int(free) if free is not None else total) * utilization)


def unmeasured_budget(
    devices: Sequence[Device], utilization: float, weights_bytes: int = 0
) -> int:
    """The budget when nothing can be measured.

    The tightest device's :func:`_unmeasured_share`, on every device;
    ``weights_bytes`` are those not loaded yet.
    """
    if all(d.is_host for d in devices):
        from .memory_plan import FreeMemory

        return int(FreeMemory.summed(devices).hi * utilization) - weights_bytes
    per_device = min(_unmeasured_share(d, utilization) for d in devices)
    return per_device * len(devices) - weights_bytes


# ---------------------------------------------------------------------------
# Warmups


def _build(tokenizer: Any, requests: Sequence[Any]) -> list[Any]:
    """Contexts of the pipeline's own type, built by its tokenizer."""
    import asyncio

    async def build() -> list[Any]:
        return [await tokenizer.new_context(r) for r in requests]

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(build())
    # A model worker builds its pipeline inside its event loop, which
    # asyncio.run refuses to nest in: run the tokenizer on its own thread.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, build()).result()


def _contexts(tokenizer: Any, lengths: Sequence[int]) -> list[Any]:
    """Text-only contexts of ``lengths`` prompt tokens each."""
    from max.pipelines.context import SamplingParams
    from max.pipelines.modeling.types import RequestID, TextGenerationRequest

    # A plain text token every vocabulary has; the tokenizer validates it.
    token = 100
    return _build(
        tokenizer,
        [
            TextGenerationRequest(
                request_id=RequestID(),
                model_name="",
                prompt=[token] * n,
                sampling_params=SamplingParams(max_new_tokens=8),
            )
            for n in lengths
        ],
    )


IMAGE_SIDES = (2048, 4096)
"""Synthetic image sides to try; the warmup takes the one with the most
tokens, the smaller on a tie. Most processors cap an image below 2048 px and
downscale or tile it to their most image tokens; the Qwen-VL family keeps
up to about 4096^2 px (16384 tokens), so 2048 px would under-fill it."""


def _png(side: int) -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    # A gradient, not a solid color: some processors crop uniform borders.
    Image.linear_gradient("L").resize((side, side)).convert("RGB").save(
        buf, format="PNG"
    )
    return buf.getvalue()


def _image_requests(n_images: int, side: int) -> Any:
    from max.pipelines.context import SamplingParams
    from max.pipelines.modeling.types import (
        ImageContentPart,
        RequestID,
        TextContentPart,
        TextGenerationRequest,
        TextGenerationRequestMessage,
        VideoContentPart,
    )

    content: list[TextContentPart | ImageContentPart | VideoContentPart] = [
        ImageContentPart() for _ in range(n_images)
    ]
    content.append(TextContentPart(text="Describe the images."))
    return TextGenerationRequest(
        request_id=RequestID(),
        model_name="",
        messages=[TextGenerationRequestMessage(role="user", content=content)],
        images=[_png(side)] * n_images,
        sampling_params=SamplingParams(max_new_tokens=8),
    )


def _image_contexts(
    tokenizer: Any,
    chunk: int,
    n: int,
    max_rows: int | None = None,
    chunked: bool = True,
) -> list[Any]:
    """Requests of the largest image, filling each of ``n`` replicas' steps.

    Images that fit fill ``chunk``, so each replica's step encodes the most
    image data a step can carry, and its prefill runs in the same step as
    the encoder, as in serving. The scheduler gives every replica a full
    budget, so each replica gets the whole ``chunk``. Under chunked prefill
    an image longer than the chunk rides alone, chunked as the scheduler
    would: its first chunk's step encodes the whole image. Without it
    (``chunked`` false) the scheduler runs a whole prompt in one step, so
    the caller passes the longest prompt as ``chunk`` and nothing is cut.
    Empty when the tokenizer takes no images, or serves text only
    (``max_images_per_request`` of 0). Each replica's step carries at most
    ``max_rows`` requests, as the scheduler's would.
    """
    limit = getattr(tokenizer, "max_images_per_request", None)
    if limit == 0:
        return []
    best: tuple[int, int] | None = None
    for side in IMAGE_SIDES:
        try:
            (probe,) = _build(tokenizer, [_image_requests(1, side)])
        except Exception as e:
            logger.debug("memory ledger: no %d px image: %s", side, e)
            continue
        if not getattr(probe, "images", None):
            continue
        if best is None or len(probe.tokens) > best[0]:
            best = (len(probe.tokens), side)
    if best is None:
        return []
    tokens, side = best
    images = max(1, chunk // max(1, tokens))
    contexts = _fill(
        tokenizer,
        n,
        images,
        limit,
        lambda k: _image_requests(k, side),
        max_rows,
    )
    return _chunked(contexts, chunk) if chunked else contexts


def _fill(
    tokenizer: Any,
    n: int,
    items: int,
    limit: int | None,
    request: Callable[[int], Any],
    max_rows: int | None = None,
) -> list[Any]:
    """``n`` replicas' worth of ``items`` media items each.

    A model that takes at most ``limit`` items per request (InternVL takes
    one) gets each replica's items split across more requests, so the step
    carries the same media. A tokenizer that names no limit but rejects a
    request of several items (one that keeps only the first item's pixels
    fails its placeholder check) takes one item per request. Each replica
    gets at most ``max_rows`` requests.
    """
    per_request = items
    if isinstance(limit, int) and limit > 0:
        per_request = min(items, limit)
    first: list[Any] = []
    if per_request > 1:
        try:
            first = _build(tokenizer, [request(per_request)])
        except Exception as e:
            logger.debug(
                "memory ledger: %d items per request rejected (%s); "
                "warming up with one per request",
                per_request,
                e,
            )
            per_request = 1
    per_replica = -(-items // per_request)
    if max_rows is not None:
        per_replica = min(per_replica, max(1, max_rows))
    rest = n * per_replica - len(first)
    return first + _build(
        tokenizer, [request(per_request) for _ in range(rest)]
    )


def _chunked(contexts: list[Any], chunk: int) -> list[Any]:
    """Limits each context's next step to ``chunk`` tokens, as the scheduler does."""
    for ctx in contexts:
        if ctx.tokens.active_length > chunk:
            ctx.tokens.chunk(chunk)
    return contexts


VIDEO_SHAPES = ((2, 3584), (16, 1280), (64, 640))
"""Synthetic clips to try, as ``(frames, side)``.

A processor bounds a clip by its total pixels, so these lay out about the
same tokens; the warmup takes the one with the most, fewest frames first,
since fewer, larger frames give the encoder its widest per-frame attention.
"""

VIDEO_FPS = 2
"""The synthetic clips' frame rate: low enough that a sampler keeps every frame."""


def _clip(frames: int, side: int) -> bytes:
    import io

    import av
    import numpy as np

    buf = io.BytesIO()
    with av.open(buf, mode="w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=VIDEO_FPS)
        stream.width = stream.height = side
        stream.pix_fmt = "yuv420p"
        ramp = np.linspace(0, 255, side, dtype=np.uint8)
        for i in range(frames):
            # A moving gradient, not a solid color: some processors crop
            # uniform borders, and a static clip may be deduplicated.
            gray = np.add.outer(ramp, ramp // 2 + i).astype(np.uint8)
            rgb = np.repeat(gray[:, :, None], 3, axis=2)
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            container.mux(stream.encode(frame))
        container.mux(stream.encode())
    return buf.getvalue()


def _video_request(n_videos: int, clip: bytes) -> Any:
    from max.pipelines.context import SamplingParams
    from max.pipelines.modeling.types import (
        ImageContentPart,
        RequestID,
        TextContentPart,
        TextGenerationRequest,
        TextGenerationRequestMessage,
        VideoContentPart,
    )

    content: list[TextContentPart | ImageContentPart | VideoContentPart] = [
        VideoContentPart() for _ in range(n_videos)
    ]
    content.append(TextContentPart(text="Describe the videos."))
    return TextGenerationRequest(
        request_id=RequestID(),
        model_name="",
        messages=[TextGenerationRequestMessage(role="user", content=content)],
        videos=[clip] * n_videos,
        sampling_params=SamplingParams(max_new_tokens=8),
    )


def _video_contexts(
    tokenizer: Any,
    chunk: int,
    n: int,
    max_rows: int | None = None,
    chunked: bool = True,
) -> list[Any]:
    """Requests of the largest clip the processor lays out, for ``n`` replicas.

    Clips that fit fill each replica's ``chunk``, as images do. Under
    chunked prefill a clip longer than the chunk is chunked as the scheduler
    would: its first chunk's step encodes the whole clip. Without it
    (``chunked`` false) nothing is cut, as with images. Empty when the
    tokenizer takes no video, or serves none (``max_videos_per_request`` of
    0). Each replica's step carries at most ``max_rows`` requests.
    """
    limit = getattr(tokenizer, "max_videos_per_request", None)
    if limit == 0:
        return []
    try:
        (text,) = _build(tokenizer, [_video_request(0, b"")])
    except Exception as e:
        logger.debug("memory ledger: no video warmup: %s", e)
        return []
    best: tuple[int, bytes] | None = None
    for frames, side in VIDEO_SHAPES:
        clip = _clip(frames, side)
        try:
            (probe,) = _build(tokenizer, [_video_request(1, clip)])
        except Exception as e:
            logger.debug("memory ledger: no %dx%d^2 clip: %s", frames, side, e)
            continue
        tokens = len(probe.tokens) - len(text.tokens)
        if tokens > 0 and (best is None or tokens > best[0]):
            best = (tokens, clip)
    if best is None:
        return []
    tokens, clip = best
    # The prompt's own text rides along, so a request of the clips fits.
    videos = max(1, (chunk - len(text.tokens)) // tokens)
    contexts = _fill(
        tokenizer,
        n,
        videos,
        limit,
        lambda k: _video_request(k, clip),
        max_rows,
    )
    return _chunked(contexts, chunk) if chunked else contexts


def _serves_images(config: Any) -> bool:
    """Whether the architecture ``config`` runs takes images, and so video.

    The check offline generation makes: a speculative pairing whose fused
    graph encodes no images declares text only, and fails a batch that
    carries any, though its tokenizer still builds one. With no
    architecture to ask, the tokenizer's probe decides.
    """
    from max.pipelines.modeling.types import InputModality

    from .registry import PIPELINE_REGISTRY

    try:
        arch = PIPELINE_REGISTRY.architecture_for_config(config)
    except Exception as e:
        logger.debug("memory ledger: no architecture for the config: %s", e)
        return True
    return arch is None or InputModality.IMAGE in arch.input_modalities


def _run(pipeline: Any, contexts: list[Any], dp: int, steps: int) -> None:
    """Prefills ``contexts`` and decodes ``steps - 1`` more tokens."""
    from max.pipelines.modeling.types.pipeline_variants.text_generation import (
        TextGenerationInputs,
    )

    kv = pipeline.kv_manager
    replicas = [i % dp for i in range(len(contexts))]
    try:
        for _ in range(steps):
            batches: list[list[Any]] = [[] for _ in range(dp)]
            for ctx, r in zip(contexts, replicas, strict=True):
                if not kv.contains(ctx):
                    kv.alloc_dummy(ctx, replica_idx=r)
                batches[r].append(ctx)
            pipeline.execute(TextGenerationInputs(batches=batches))
        # An overlapped pipeline returns a step late: one more drains it.
        if type(pipeline).__name__.startswith("Overlap"):
            pipeline.execute(
                TextGenerationInputs(batches=[[] for _ in range(dp)])
            )
    finally:
        for ctx in contexts:
            if kv.contains(ctx):
                kv.release(ctx)
            pipeline.release(ctx.request_id)


def _cached(tokenizer: Any, prompt_len: int, cached: int, n: int) -> list[Any]:
    """``n`` contexts of ``prompt_len`` tokens, ``cached`` of them cached.

    Their next forward attends over the cached tokens.
    """
    # Only the first context goes through the tokenizer: building every row
    # at a long max_length (512 rows of 262K tokens) costs tens of seconds of
    # boot, and a row's forward reads only its lengths.
    first = _contexts(tokenizer, [prompt_len])[0]
    contexts = [first, *(_clone(first) for _ in range(n - 1))]
    if cached > 0:
        for ctx in contexts:
            ctx.tokens.skip_processing(cached)
    return contexts


def _clone(ctx: Any) -> Any:
    """A copy of ``ctx`` with its own request ID and processing state.

    The copies share the prompt's token array; each gets a token buffer of
    its own over it, so advancing one row leaves the others where they are.
    """
    import copy

    from max.pipelines.context.tokens import Range
    from max.pipelines.modeling.types import RequestID

    clone = copy.copy(ctx)
    clone.request_id = RequestID()
    tokens = copy.copy(ctx.tokens)
    for name in ("_processing_range", "_completion_range"):
        r = getattr(tokens, name)
        setattr(tokens, name, Range(r.start, r.end))
    clone.tokens = tokens
    return clone


def run_warmups(
    pipeline: Any,
    tokenizer: Any,
    max_batch_size: int,
    max_length: int,
    devices: Sequence[Device] = (),
) -> dict[str, list[int]]:
    """Runs the shapes whose workspaces are largest, once each.

    The widest prefill from an empty cache and at the tail of the longest
    sequence (a chunk's workspaces also grow with the context it attends
    over), the largest decode batch at the longest context, and, for a
    model that takes images, a prefill chunk filled with the largest images
    (the vision encoder and the prefill run in that one step), and for one
    that takes video, the first chunk of the largest clip. Every
    request's pages are the null page, so any shape fits.

    Returns:
        Per shape (``"prefill"``, ``"prefill_long"``, ``"decode"``, and
        ``"image"`` and ``"video"`` for a vision model), the
        peak each of ``devices`` reached above its live bytes when the
        shape ended: its transients, without what it kept.
    """
    config = pipeline.pipeline_config
    dp = max(1, config.model.data_parallel_degree)
    longest = max(1, max_length - 4)
    chunk = max(1, min(config.runtime.max_batch_input_tokens, longest))
    # Without chunked prefill the scheduler runs a whole prompt in one step,
    # however long: the widest prefill is the longest prompt itself.
    chunked = bool(config.runtime.enable_chunked_prefill)
    prefill = chunk if chunked else longest
    shapes = [
        (
            "prefill",
            "prefill from an empty cache",
            _contexts(tokenizer, [prefill] * dp),
            1,
        ),
    ]
    if chunked:
        shapes.append(
            (
                "prefill_long",
                "prefill at the longest context",
                _cached(tokenizer, longest, longest - chunk, dp),
                1,
            )
        )
    shapes += [
        (
            "decode",
            "decode at the largest batch and longest context",
            _cached(
                tokenizer,
                longest,
                longest - 1,
                max(1, max_batch_size // dp) * dp,
            ),
            3,
        ),
    ]
    serves_media = _serves_images(config)
    rows = max(1, max_batch_size // dp)
    # Media fill each replica's widest prefill: a chunk, or without chunked
    # prefill the longest prompt, which one step then encodes whole.
    images = (
        _image_contexts(tokenizer, prefill, dp, rows, chunked)
        if serves_media
        else []
    )
    if images:
        shapes.append(
            ("image", "prefill a chunk of images", images, 1),
        )
    elif (
        serves_media and getattr(tokenizer, "max_images_per_request", None) != 0
    ):
        # The architecture takes images, but its tokenizer built no image
        # request (the reason is logged at debug).
        logger.warning(
            "memory ledger: the model takes images, but no image warmup could "
            "be built; the KV cache is sized without the vision encoder's "
            "transients"
        )
    videos = (
        _video_contexts(tokenizer, prefill, dp, rows, chunked)
        if serves_media
        else []
    )
    if videos:
        shapes.append(
            ("video", "prefill the largest video", videos, 1),
        )
    peaks: dict[str, list[int]] = {}
    for key, name, contexts, steps in shapes:
        for d in devices:
            d.reset_memory_manager_peaks()
        t0 = time.monotonic()
        _run(pipeline, contexts, dp, steps=steps)
        # Above what is live at the shape's end, not its start: what a shape
        # allocates and keeps (an arena built on first use, an overlapped
        # step's outputs) is counted once, by the caller, as resident.
        peaks[key] = [
            max(
                0,
                int(d.memory_manager_stats["peak_live_bytes"]) - _live_bytes(d),
            )
            for d in devices
        ]
        logger.info(
            "memory ledger: warmup %s (%d rows, chunk %d, context %d) took "
            "%.1fs",
            name,
            len(contexts),
            chunk,
            longest,
            time.monotonic() - t0,
        )
    return peaks


def _is_oom(e: BaseException) -> bool:
    return (
        "out of memory" in str(e).lower() or "oom" in type(e).__name__.lower()
    )


# ---------------------------------------------------------------------------
# Sizing


@dataclass
class Ledger:
    """A pipeline's measurement, kept to hold graph capture to it."""

    devices: list[DeviceLedger]
    live_before_capture: list[int] = dataclasses.field(default_factory=list)


def _measure(
    devices: Sequence[Device],
    resident: Sequence[int],
    live_at_resident: Sequence[int],
    peaks: dict[str, list[int]],
    captures: bool,
) -> list[DeviceLedger]:
    """Each device's ledger after the warmups.

    ``resident`` was read with ``live_at_resident`` live; what the warmups
    left live past that is resident too, since ``peaks`` exclude it.
    """
    out = []
    for i, d in enumerate(devices):
        d.synchronize()
        stats = dict(d.memory_manager_stats)
        kept = max(0, int(stats["live_bytes"]) - live_at_resident[i])
        out.append(
            DeviceLedger(
                total_bytes=int(d.stats["total_memory"]),
                resident_bytes=resident[i] + kept,
                eager_bytes=max((p[i] for p in peaks.values()), default=0),
                capture_bytes=peaks.get("decode", [0] * len(devices))[i]
                if captures
                else 0,
                driver_fallbacks=stats["driver_fallbacks"],
            )
        )
    return out


def size_kv_cache(pipeline: Any, plan: MemoryPlan) -> MemoryPlan:
    """Measures the warmups against the null cache and installs the real one.

    The pipeline is fully built against a
    :class:`~max.pipelines.kv_cache.NullKVCache`. Returns the plan the real
    cache was sized with: its budget, the clamped ``max_length``, and the
    batch size the warmups fit at.

    Args:
        pipeline: A text-generation pipeline exposing ``kv_manager``,
            ``tokenizer``, ``pipeline_config``, ``_devices``,
            ``_pipeline_model`` and ``install_kv_manager``.
        plan: The plan the pipeline was built with.
    """
    from max.pipelines.kv_cache import (
        NullKVCache,
        kv_cache_memory_size,
        load_kv_manager,
        max_seq_len_fitting_in_cache,
    )

    from .memory_plan import vision_cache_plan

    config = pipeline.pipeline_config
    devices: list[Device] = list(pipeline._devices)
    model = pipeline._pipeline_model
    params = model.kv_params
    utilization = config.model.kv_cache.device_memory_utilization
    is_di = config.runtime.is_disaggregated
    model_name = config.model.model_name
    max_length = plan.planned_max_length
    assert max_length is not None
    batch = plan.planned_max_batch_size
    user_batch = config.runtime.max_batch_size is not None
    assert isinstance(pipeline.kv_manager, NullKVCache)
    # Held by value: a reference to the null cache would keep its slab
    # alive past the swap.
    null_slab_bytes = pipeline.kv_manager.slab_bytes

    chunk = max(1, min(config.runtime.max_batch_input_tokens, max_length))
    min_cache_bytes = kv_cache_memory_size(
        params=params,
        available_cache_memory=1 << 50,
        max_batch_size=1,
        max_seq_len=chunk,
        is_di_enabled=is_di,
        model_name=model_name,
    )

    def install(
        cache_bytes: int, batch_size: int, length: int, vision: Any
    ) -> MemoryPlan:
        new_plan = dataclasses.replace(
            plan,
            planned_max_batch_size=batch_size,
            planned_max_length=length,
            available_cache_memory=cache_bytes,
            vision_cache_plan=vision,
            planned_max_batch_total_tokens=(
                length
                if plan.planned_max_batch_total_tokens
                == plan.planned_max_length
                and config.runtime.max_batch_total_tokens is None
                else plan.planned_max_batch_total_tokens
            ),
        )
        model.memory_plan = new_plan
        pipeline.kv_manager.shutdown()
        # Drop the null cache's slab before the real one claims the memory.
        pipeline.install_kv_manager(None, new_plan)
        pipeline.install_kv_manager(
            load_kv_manager(
                params=params,
                max_batch_size=batch_size,
                max_seq_len=length,
                session=pipeline.session,
                available_cache_memory=cache_bytes,
                is_di_enabled=is_di,
                model_name=model_name,
                max_num_input_tokens=config.runtime.max_num_input_tokens,
            ),
            new_plan,
        )
        return new_plan

    def carve(cache_bytes: int) -> tuple[int, Any]:
        """The KV bytes and vision cache plan ``cache_bytes`` leaves."""
        # One buffer per device; a device caps its largest allocation.
        cache_bytes = min(
            cache_bytes, sum(d.max_single_alloc_size for d in devices)
        )
        vision_bytes, vision = vision_cache_plan(
            config, plan.vision_cache_plan, cache_bytes, len(devices)
        )
        return cache_bytes - vision_bytes, vision

    def fits(cache_bytes: int) -> bool:
        """Whether ``cache_bytes`` leaves the KV cache one prefill chunk."""
        try:
            return carve(cache_bytes)[0] >= min_cache_bytes
        except ValueError:
            # Too small for one vision cache block.
            return False

    def finish(cache_bytes: int, batch_size: int) -> MemoryPlan:
        kv_bytes, vision = carve(cache_bytes)
        if kv_bytes < min_cache_bytes:
            raise RuntimeError(
                f"The KV cache ({to_human_readable_bytes(kv_bytes)}, after "
                "the largest single allocation cap and the vision encoder "
                "cache) is smaller than the "
                f"{to_human_readable_bytes(min_cache_bytes)} one prefill "
                "chunk needs. Try a higher --device-memory-utilization, a "
                "smaller --max-batch-input-tokens, or more devices."
            )
        capacity = max_seq_len_fitting_in_cache(
            params=params,
            available_cache_memory=kv_bytes,
            is_di_enabled=is_di,
            model_name=model_name,
        )
        length = max_length
        if capacity is not None and capacity < max_length:
            logger.warning(
                "Clamping max_length from %d to %d: the measured KV cache (%s) "
                "holds no longer request",
                max_length,
                capacity,
                to_human_readable_bytes(kv_bytes),
            )
            length = capacity
        logger.info("cache_memory: %s", to_human_readable_bytes(kv_bytes))
        return install(kv_bytes, batch_size, length, vision)

    if not measurable(devices):
        cache_bytes = unmeasured_budget(devices, utilization)
        logger.warning(
            "memory ledger: the memory manager reports no statistics, so the "
            "KV cache is sized with a fixed %s transient reserve per GPU",
            to_human_readable_bytes(UNMEASURED_TRANSIENT_BYTES),
        )
        if cache_bytes <= 0:
            raise RuntimeError(
                "The model leaves no memory for a KV cache at "
                f"device_memory_utilization={utilization}."
            )
        return finish(cache_bytes, batch)

    # Collect garbage first: an earlier pipeline in this process that is no
    # longer referenced (but held in a cycle) would otherwise count as
    # resident and leave this one no cache.
    gc.collect()
    # Everything but the null slab stays: the real cache takes its place.
    resident = [_used_bytes(d) - null_slab_bytes for d in devices]
    live_at_resident = [_live_bytes(d) for d in devices]
    captures = bool(config.runtime.device_graph_capture)
    while True:
        error: BaseException | None = None
        peaks: dict[str, list[int]] = {}
        t0 = time.monotonic()
        try:
            peaks = run_warmups(
                pipeline, pipeline.tokenizer, batch, max_length, devices
            )
        except Exception as e:
            if not _is_oom(e):
                raise
            error = e
        ledgers = _measure(devices, resident, live_at_resident, peaks, captures)
        cache_bytes = cache_budget(ledgers, utilization)
        logger.info(
            "memory ledger: warmups at max_batch_size=%d took %.1fs",
            batch,
            time.monotonic() - t0,
        )
        if error is None and fits(cache_bytes):
            break
        detail = (
            f"{type(error).__name__}: {error}"
            if error is not None
            else (
                f"the measured transients leave "
                f"{to_human_readable_bytes(cache_bytes)} for the KV cache, "
                f"less than the {to_human_readable_bytes(min_cache_bytes)} "
                "one prefill chunk needs"
            )
        )
        if user_batch or batch <= 1:
            log_ledger(ledgers, utilization)
            raise RuntimeError(
                f"The model does not fit at max_batch_size={batch}: {detail}. "
                "Try a smaller --max-batch-size, a smaller "
                "--max-batch-input-tokens, a higher "
                "--device-memory-utilization, or more devices."
            )
        logger.warning(
            "memory ledger: max_batch_size=%d does not fit (%s); trying %d",
            batch,
            detail,
            batch // 2,
        )
        batch //= 2
        pipeline.install_kv_manager(None, plan)
        pipeline.install_kv_manager(
            NullKVCache(
                params=params,
                max_batch_size=batch,
                max_seq_len=max_length,
                max_num_input_tokens=config.runtime.max_num_input_tokens,
            ),
            dataclasses.replace(plan, planned_max_batch_size=batch),
        )
    for i, m in enumerate(ledgers):
        if m.driver_fallbacks:
            logger.warning(
                "memory ledger: device %d fell back to the driver %d times; "
                "those bytes are counted",
                i,
                m.driver_fallbacks,
            )
    log_ledger(ledgers, utilization)

    new_plan = finish(cache_bytes, batch)
    pipeline.memory_ledger = Ledger(
        devices=ledgers,
        live_before_capture=[_live_bytes(d) for d in devices],
    )
    return new_plan


def check_capture(pipeline: Any) -> None:
    """Holds graph capture to the share the decode warmup measured.

    The graphs a model captures share one memory pool and replay one at a
    time, so what capture keeps is about the largest decode's workspaces,
    claimed on top of the eager forwards' (which still run, beside the
    kept pool). More than that share plus the margin means the KV cache
    took memory the graphs need, which would surface later as an
    out-of-memory mid-serve.

    Raises:
        RuntimeError: If any device's capture kept more than its claim.
    """
    ledger: Ledger | None = getattr(pipeline, "memory_ledger", None)
    if ledger is None or not ledger.live_before_capture:
        return
    margin = margin_bytes()
    for i, d in enumerate(pipeline._devices):
        kept = _live_bytes(d) - ledger.live_before_capture[i]
        claim = ledger.devices[i].capture_bytes + margin
        if kept > claim:
            raise RuntimeError(
                f"Graph capture kept {to_human_readable_bytes(kept)} on device "
                f"{i}, more than the {to_human_readable_bytes(claim)} the "
                "decode warmup measured plus the margin. Raise "
                f"{MARGIN_ENV} or lower --device-memory-utilization."
            )
    logger.info("memory ledger: graph capture stayed inside its measured share")
