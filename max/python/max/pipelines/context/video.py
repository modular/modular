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

"""Shared video decoding for pipelines that consume untrusted video input.

Every model that samples frames from a video needs the same things: open the
container safely, learn its shape, choose which frames to keep, decode only
those, and stay inside a memory bound while doing it. Only the *choice* of
frames is model-specific, and it is expressed here as arguments rather than as
a second decoder.

Re-implementing the rest per model is how the predecessors of this module
ended up with different bugs: one materialized the whole clip before
subsampling (a few-MB low-entropy upload expands to hundreds of GB of
rasters), one trusted a container header that lied about its frame count, and
one selected frames by counting them -- which silently returns the wrong
frames on a damaged stream. Each fix landed in one copy. They live here now so
there is one copy to fix.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import tempfile
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import IO, Literal

import av
import numpy as np
import numpy.typing as npt
from PIL import Image

from .exceptions import InputError, MediaBudgetExceeded, MediaDecodeError

logger = logging.getLogger(__name__)

__all__ = [
    "SampledVideo",
    "VideoSource",
    "VideoStreamInfo",
    "decode_video_frames",
    "open_video_container",
    "plan_frame_indices",
    "probe_video",
    "validate_video",
]

# TODO(SDLC-4121): Remove once the pinned `av` build vendors an FFmpeg
# >= 8.1.2, which fixes CVE-2026-8461 (a heap out-of-bounds write in the
# MagicYUV decoder).
_BLOCKED_VIDEO_CODECS = frozenset({"magicyuv"})

# Fallback for the very unusual container that declares no frame rate; real
# clips essentially always carry one. Only affects FPS-based sampling and
# caller-computed timestamps.
_DEFAULT_FPS = 30.0

# Decoded frames are RGB, one byte per channel.
_BYTES_PER_PIXEL = 3

# Thread count for the parallel sampled decode, clamped to [1, 64]. Override
# with ``MODULAR_VIDEO_DECODE_THREADS``. Threading helps because each worker
# seeks to its own span, so a long clip is not walked end-to-end serially.
_DEFAULT_DECODE_THREADS = 16

# Below this many sampled frames the thread pool and its temp file cost more
# than the parallelism saves.
_MIN_FRAMES_PER_THREAD = 8

# Tolerance for "this frame is at the target time", in frame intervals.
_TARGET_TOLERANCE = 0.5

# A video to decode: encoded bytes, a filesystem path, or a ``data:`` URI.
# Decoding may open the source more than once, so it must be re-openable --
# which is why this is not a file object.
VideoSource = str | bytes

Rounding = Literal["round", "floor"]

# How a target frame rate selects frames.
#
# ``uniform`` spreads the sampled count evenly across the clip with
# ``linspace``. ``walk`` keeps frame 0, then each next frame at least
# ``1 / target_fps`` seconds after the last kept one, plus the final frame.
# They differ on what they guarantee: ``uniform`` guarantees the count,
# ``walk`` guarantees the spacing and the final frame. Models that evaluate on
# benchmarks whose answer lives in the last frame need ``walk``.
Sampling = Literal["uniform", "walk"]

_WALK_EPS = 1e-4


@dataclass(frozen=True)
class VideoStreamInfo:
    """Header facts about a video's first video stream.

    Args:
        fps: Frame rate the container declares, or ``None`` when it declares
            none. Callers that need a number apply their own fallback.
        num_frames: Number of decodable frames. Best-effort from the header
            until a decode corrects it; ``0`` when the header offers nothing.
        width: Frame width in pixels (``0`` when unreported).
        height: Frame height in pixels (``0`` when unreported).
    """

    fps: float | None
    num_frames: int
    width: int
    height: int

    @property
    def frame_bytes(self) -> int:
        """Decoded size of one RGB frame, in bytes."""
        return self.width * self.height * _BYTES_PER_PIXEL


@dataclass(frozen=True)
class SampledVideo:
    """The frames kept from a video, and the source they came from.

    Frames are held as ``uint8`` ``(H, W, C)`` RGB arrays -- the form PyAV
    produces -- and converted on request. ``frame.to_ndarray("rgb24")`` is
    pixel-identical to ``frame.to_image().convert("RGB")``, so callers that
    want images pay only for the wrapper, not a second conversion.

    Args:
        frames: The sampled frames, in ``indices`` order.
        indices: Source frame index of each entry in ``frames``.
        info: Stream metadata, with ``num_frames`` corrected to the count the
            decode actually produced.
    """

    frames: list[npt.NDArray[np.uint8]]
    indices: list[int]
    info: VideoStreamInfo

    @property
    def images(self) -> list[Image.Image]:
        """The sampled frames as RGB :class:`PIL.Image.Image` objects."""
        return [Image.fromarray(frame, mode="RGB") for frame in self.frames]

    @property
    def array(self) -> npt.NDArray[np.uint8]:
        """The sampled frames stacked into a ``(T, H, W, C)`` uint8 array."""
        return np.stack(self.frames, axis=0)


def open_video_container(
    source: str | IO[bytes],
) -> av.container.InputContainer:
    """Opens a video container for reading via PyAV, rejecting known-vulnerable codecs.

    Video-decoding pipelines should open containers through this function
    rather than calling ``av.open`` directly, so the codec blocklist applies
    uniformly everywhere PyAV decodes untrusted video input.

    Args:
        source: A file path or file-like object containing the encoded video.

    Returns:
        The opened input container.

    Raises:
        MediaDecodeError: If any video stream uses a blocked codec.
    """
    container = av.open(source, mode="r")
    assert isinstance(container, av.container.InputContainer)
    for stream in container.streams.video:
        codec_name = stream.codec_context.name
        if codec_name in _BLOCKED_VIDEO_CODECS:
            container.close()
            raise MediaDecodeError(f"Unsupported video codec {codec_name!r}.")
    return container


@contextmanager
def _wrap_decode_errors() -> Iterator[None]:
    """Turns PyAV/IO failures on untrusted input into :class:`MediaDecodeError`.

    PyAV reports malformed, truncated, and unreadable media as
    :class:`av.error.FFmpegError`, and a bad base64 payload as
    :class:`ValueError`; both become :class:`MediaDecodeError`, so callers
    never see a backend exception type. Anything else (``MemoryError``,
    programming errors, thread-pool failures) is a server fault and propagates
    as a 500. :class:`InputError` passes through unchanged to keep the specific
    message.
    """
    try:
        yield
    except InputError:
        raise
    except (av.error.FFmpegError, ValueError) as e:
        raise MediaDecodeError("invalid or unreadable video content") from e


def _video_bytes(source: VideoSource) -> bytes | None:
    """Returns the encoded bytes of ``source``, or ``None`` for a path."""
    if isinstance(source, bytes):
        return source
    if source.startswith("data:"):
        return base64.b64decode(source.split(",", 1)[-1])
    return None


def _open(source: VideoSource) -> av.container.InputContainer:
    """Opens ``source``, decoding a ``data:`` URI or wrapping raw bytes.

    Each call builds a fresh stream, so the caller can open the same source
    for the metadata probe, the counting decode, and the sampling decode.
    """
    raw = _video_bytes(source)
    if raw is not None:
        return open_video_container(io.BytesIO(raw))
    assert isinstance(source, str)
    return open_video_container(source)


def _decode_threads() -> int:
    """Thread count for the parallel sampled decode, clamped to ``[1, 64]``."""
    raw = os.environ.get("MODULAR_VIDEO_DECODE_THREADS", "")
    if raw:
        try:
            return max(1, min(64, int(raw)))
        except ValueError:
            pass
    return _DEFAULT_DECODE_THREADS


def _declared_frame_count(
    stream: av.video.stream.VideoStream,
    container: av.container.InputContainer,
    fps: float,
) -> int:
    """Best-effort frame count from container metadata, or ``0`` if unknown.

    Falls back through the stream duration and then the container duration,
    because common containers (MPEG-TS, fragmented MP4) carry no frame count.
    ``0`` tells the caller to count with a decode instead.
    """
    total = stream.frames or 0
    if total <= 0 and stream.duration is not None and stream.time_base:
        total = round(float(stream.duration * stream.time_base) * fps)
    if total <= 0 and container.duration is not None:
        total = round((container.duration / av.time_base) * fps)
    return max(total, 0)


def probe_video(source: VideoSource) -> VideoStreamInfo:
    """Reads a video's header without decoding any frame.

    Args:
        source: The video to probe.

    Returns:
        The stream metadata. ``num_frames`` is the header's best-effort count
        and may be ``0`` or simply wrong; :func:`decode_video_frames` corrects
        it against the frames a decode actually produces.

    Raises:
        MediaDecodeError: If the source is not a readable video.
    """
    with _wrap_decode_errors():
        with _open(source) as container:
            if not container.streams.video:
                raise MediaDecodeError("invalid or unreadable video content")
            stream = container.streams.video[0]
            rate = stream.average_rate or stream.base_rate
            fps = float(rate) if rate else None
            return VideoStreamInfo(
                fps=fps,
                num_frames=_declared_frame_count(
                    stream, container, fps or _DEFAULT_FPS
                ),
                width=stream.width or 0,
                height=stream.height or 0,
            )


def validate_video(
    source: VideoSource, max_decoded_bytes: int | None = None
) -> None:
    """Checks a video's header, decoding no frame.

    Intended for trust boundaries (the serving layer) that want an unreadable
    or oversized video rejected as a client error before it reaches a model
    worker. The video is decoded exactly once, by whoever consumes it, so
    damage inside the stream surfaces there rather than here.

    Args:
        source: The video to validate.
        max_decoded_bytes: Decoded-byte budget available to the video. A frame
            is allocated at full size before any check can run on it, so the
            header dimensions are checked against this before decoding.

    Raises:
        MediaDecodeError: If the source is not a readable video.
        MediaBudgetExceeded: If one frame would not fit in
            ``max_decoded_bytes``.
    """
    _budget_frame_limit(probe_video(source), max_decoded_bytes)


def _budget_frame_limit(
    info: VideoStreamInfo, max_decoded_bytes: int | None
) -> int | None:
    """How many frames of this video fit in ``max_decoded_bytes``.

    A video's *encoded* size says nothing about how much it decodes to: a
    few-MB clip of low-entropy content expands to hundreds of GB of raster.
    Charging the decode against the same per-request byte budget as everything
    else -- estimated from the header as ``width * height * 3`` per frame, so
    it is known before a single frame is allocated -- bounds that without a
    separate frame-count limit, which is a poor proxy anyway (512 frames of 4K
    is ~12 GB).

    Returns:
        The frame limit, or ``None`` when ``max_decoded_bytes`` is ``None``.

    Raises:
        MediaBudgetExceeded: If a single frame alone exceeds the budget.
        MediaDecodeError: If the header reports no frame dimensions (the
            footprint cannot be bounded).
    """
    if max_decoded_bytes is None:
        return None
    frame_bytes = info.frame_bytes
    if frame_bytes <= 0:
        raise MediaDecodeError("video does not declare its frame dimensions")
    limit = max_decoded_bytes // frame_bytes
    if limit < 1:
        raise MediaBudgetExceeded(
            f"video frame decodes to {frame_bytes} bytes, more than the "
            f"maximum request size of {max_decoded_bytes} bytes"
        )
    return limit


def _uniform_indices(
    total: int,
    count: int,
    *,
    start: int,
    end: int,
    rounding: Rounding,
) -> list[int]:
    """Spreads ``count`` samples evenly across ``[start, end]``."""
    positions = np.linspace(start, end, count)
    # Upstream references disagree here and the choice shifts which source
    # frame each sample lands on, so it stays the caller's to make.
    if rounding == "round":
        positions = positions.round()
    return positions.astype(int).clip(0, max(total - 1, 0)).tolist()


def _walk_indices(total: int, fps: float, target_fps: float) -> list[int]:
    """Keeps frame 0, frames ``1 / target_fps`` apart, and the final frame.

    Unlike an evenly spread sample this guarantees the spacing rather than the
    count, and always ends on the last frame -- which matters for benchmarks
    whose question is the final frame.
    """
    if total <= 0:
        return []
    interval = 1.0 / target_fps
    indices: list[int] = []
    prev_ts = -np.inf
    while True:
        if not indices:
            target = 0
        else:
            target_ts = prev_ts + interval - _WALK_EPS
            target = max(int(np.ceil(target_ts * fps)), indices[-1] + 1)
        if target >= total:
            break
        indices.append(target)
        prev_ts = target / fps
    last = total - 1
    if indices and indices[-1] != last and last / fps - prev_ts > _WALK_EPS:
        indices.append(last)
    return indices or [0]


def _plan_indices(
    total: int,
    fps: float,
    *,
    num_frames: int | None,
    target_fps: float | None,
    sampling: Sampling,
    start_frame: int | None,
    end_frame: int | None,
    min_frames: int,
    max_frames: int | None,
    budget_frames: int | None,
    rounding: Rounding,
) -> list[int]:
    """Chooses which source frames to keep, within every applicable bound."""
    start = 0 if start_frame is None else max(start_frame, 0)
    end = total - 1 if end_frame is None else min(end_frame, total - 1)
    end = max(end, start)
    span = end - start + 1

    if sampling == "walk" and num_frames is None:
        indices = _walk_indices(span, fps, target_fps or fps)
        indices = [start + i for i in indices]
        # The walk sets its own count, so the caps apply by thinning it
        # evenly -- keeping the first and last entries, which is the property
        # the walk exists to guarantee.
        cap = _effective_cap(
            len(indices), min_frames, max_frames, budget_frames
        )
        if cap < len(indices):
            keep = _uniform_indices(
                len(indices),
                cap,
                start=0,
                end=len(indices) - 1,
                rounding="round",
            )
            indices = [indices[i] for i in keep]
        return indices

    if num_frames is not None:
        count = num_frames
    elif target_fps is not None:
        count = round(span * min(target_fps, fps) / fps)
    else:
        count = span
    count = _effective_cap(count, min_frames, max_frames, budget_frames)
    count = min(count, span)
    return _uniform_indices(
        total, max(count, 1), start=start, end=end, rounding=rounding
    )


def plan_frame_indices(
    total_frames: int,
    fps: float,
    *,
    num_frames: int | None = None,
    target_fps: float | None = None,
    sampling: Sampling = "uniform",
    start_frame: int | None = None,
    end_frame: int | None = None,
    min_frames: int = 1,
    max_frames: int | None = None,
    rounding: Rounding = "round",
) -> list[int]:
    """Predicts which frames :func:`decode_video_frames` would keep.

    Same policy arguments, no decoding, so a caller can size the result before
    committing to it -- for instance to reject a request from the container
    header alone rather than after paying for the decode. The decoded-byte
    budget is not applied here, so this is an upper bound on the sample.

    Args:
        total_frames: Number of frames in the source.
        fps: The source frame rate.
        num_frames: See :func:`decode_video_frames`.
        target_fps: See :func:`decode_video_frames`.
        sampling: See :func:`decode_video_frames`.
        start_frame: See :func:`decode_video_frames`.
        end_frame: See :func:`decode_video_frames`.
        min_frames: See :func:`decode_video_frames`.
        max_frames: See :func:`decode_video_frames`.
        rounding: See :func:`decode_video_frames`.

    Returns:
        The source frame indices that would be sampled.
    """
    return _plan_indices(
        total_frames,
        fps,
        num_frames=num_frames,
        target_fps=target_fps,
        sampling=sampling,
        start_frame=start_frame,
        end_frame=end_frame,
        min_frames=min_frames,
        max_frames=max_frames,
        budget_frames=None,
        rounding=rounding,
    )


def _effective_cap(
    count: int,
    min_frames: int,
    max_frames: int | None,
    budget_frames: int | None,
) -> int:
    """Applies the floor and every ceiling to a requested sample count."""
    count = max(count, min_frames)
    if max_frames is not None:
        count = min(count, max_frames)
    if budget_frames is not None:
        count = min(count, budget_frames)
    return max(count, 1)


def _frame_rgb24(frame: av.VideoFrame) -> npt.NDArray[np.uint8]:
    """Converts a decoded frame to an rgb24 array.

    ``to_ndarray`` is typed as a dtype union across all pixel formats; for
    rgb24 the result is always uint8, and ``astype(copy=False)`` narrows the
    static type without copying.
    """
    return frame.to_ndarray(format="rgb24").astype(np.uint8, copy=False)


def _decode_span(
    source: str | bytes,
    targets: list[float],
    fps: float,
) -> tuple[list[npt.NDArray[np.uint8] | None], av.VideoFrame | None, int]:
    """Decodes the frames at ``targets`` seconds, selecting by timestamp.

    Selecting by presentation timestamp is what makes this correct on bad
    inputs: an index-counting decoder assumes it landed where it asked, so on a
    damaged stream it returns frames from the wrong position, silently and
    non-deterministically. A frame that carries its own timestamp cannot be
    mislabeled.

    The same property covers undecodable tails and headers that overcount
    frames (VFR clips, rounded durations, truncated downloads): a stream that
    ends before a target time simply stops producing frames, and the caller
    substitutes, rather than raising.

    Returns:
        The frame for each target (``None`` where the stream ended first), the
        last frame decoded in this span, and how many frames were seen.
    """
    out: list[npt.NDArray[np.uint8] | None] = [None] * len(targets)
    last_obj: av.VideoFrame | None = None
    seen = 0
    half = _TARGET_TOLERANCE / fps
    opened = io.BytesIO(source) if isinstance(source, bytes) else source
    with open_video_container(opened) as container:
        stream = container.streams.video[0]
        # AUTO threading parallelises slice- and frame-level codec work across
        # cores, which dominates decode wall-time on H.264.
        stream.thread_type = "AUTO"
        ctx = stream.codec_context
        tb = stream.time_base or Fraction(1, 90000)
        # Containers such as MPEG-TS start at a nonzero PTS; targets are
        # relative to the stream origin.
        origin = (stream.start_time or 0) * tb
        # Give the reorder buffer room: only skip non-reference frames when the
        # next target is comfortably far away.
        lead = max(0.35, 10.0 / fps)
        try:
            if targets and targets[0] > 0:
                container.seek(
                    int((targets[0] + origin) / tb),
                    stream=stream,
                    backward=True,
                )
        except av.error.FFmpegError:
            pass  # unseekable/odd container: decode from the start
        want = 0
        try:
            for frame in container.decode(stream):
                seen += 1
                if frame.pts is None:
                    continue
                t = float(frame.pts * tb - origin)
                arr: npt.NDArray[np.uint8] | None = None
                while want < len(targets) and t >= targets[want] - half:
                    if arr is None:
                        arr = _frame_rgb24(frame)
                    out[want] = arr
                    want += 1
                last_obj = frame
                if want >= len(targets):
                    break
                # Gap accelerator: non-reference frames in the gap are never
                # emitted, so skipping their decode is safe.
                ctx.skip_frame = (
                    "DEFAULT" if t >= targets[want] - lead else "NONREF"
                )
        except av.error.FFmpegError:
            pass  # damaged region or truncated tail: the caller substitutes
    return out, last_obj, seen


def _count_frames(source: VideoSource) -> int:
    """Counts decodable frames with a decode that retains no pixels.

    Stops at the first damaged or truncated region rather than raising: the
    answer wanted here is how many frames can actually be decoded, which is
    precisely what a stream that dies partway through has fewer of than its
    header claims.
    """
    total = 0
    with _open(source) as container:
        try:
            for _ in container.decode(video=0):
                total += 1
        except av.error.FFmpegError:
            pass
    return total


def _fill_gaps(
    out: list[npt.NDArray[np.uint8] | None],
    last_obj: av.VideoFrame | None,
    source: VideoSource,
) -> tuple[list[npt.NDArray[np.uint8]], int]:
    """Substitutes the nearest decoded frame for any target that had none.

    A target past the end of the decodable stream (a truncated download or a
    damaged tail) takes the last frame that did decode, and an interior gap
    takes the nearest earlier one, so a partly-damaged clip still yields a
    usable sample instead of failing the request.

    Returns:
        The gap-free frames, and how many had to be substituted.
    """
    missing = sum(1 for entry in out if entry is None)
    if not missing:
        return [entry for entry in out if entry is not None], 0

    last = _frame_rgb24(last_obj) if last_obj is not None else None
    if missing == len(out) and last is None:
        # Nothing decodable in this span: fall back to the stream's first
        # decodable frame.
        with _open(source) as container:
            for frame in container.decode(video=0):
                last = _frame_rgb24(frame)
                break
    if missing == len(out) and last is None:
        raise MediaDecodeError("video contains no decodable frames")

    tail = len(out)
    while tail > 0 and out[tail - 1] is None:
        tail -= 1
    for i in range(tail, len(out)):
        out[i] = last
    first_present = next(entry for entry in out if entry is not None)
    prev: npt.NDArray[np.uint8] = first_present
    filled: list[npt.NDArray[np.uint8]] = []
    for entry in out:
        prev = entry if entry is not None else prev
        filled.append(prev)
    return filled, missing


@contextmanager
def _as_path(source: VideoSource) -> Iterator[str]:
    """Yields a filesystem path for ``source``, writing a temp file if needed.

    One shared file lets every decode worker open its own container and seek to
    its own span, and replaces N in-memory copies of the encoded bytes, which
    reach hundreds of MiB.
    """
    raw = _video_bytes(source)
    if raw is None:
        assert isinstance(source, str)
        yield source
        return
    fd, path = tempfile.mkstemp(suffix=".video")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
        yield path
    finally:
        os.unlink(path)


def _decode_at_indices(
    source: VideoSource, indices: list[int], fps: float, threads: int
) -> tuple[list[npt.NDArray[np.uint8]], int]:
    """Decodes the frames at ``indices``, in parallel spans when worthwhile.

    Returns:
        The frames, and how many targets the stream ended before reaching --
        the signal that the declared frame count was too high.
    """
    if not indices:
        return [], 0

    n_threads = max(1, min(threads, len(indices) // _MIN_FRAMES_PER_THREAD))
    if n_threads == 1:
        raw = _video_bytes(source)
        out, last_obj, _ = _decode_span(
            raw if raw is not None else source,
            [idx / fps for idx in indices],
            fps,
        )
        return _fill_gaps(out, last_obj, source)

    chunk_size = -(-len(indices) // n_threads)  # ceil
    chunks = [
        indices[i : i + chunk_size] for i in range(0, len(indices), chunk_size)
    ]
    with _as_path(source) as path:

        def decode_chunk(
            chunk: list[int],
        ) -> tuple[
            list[npt.NDArray[np.uint8] | None], av.VideoFrame | None, int
        ]:
            return _decode_span(path, [idx / fps for idx in chunk], fps)

        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            results = list(pool.map(decode_chunk, chunks))

    frames: list[npt.NDArray[np.uint8]] = []
    missing_total = 0
    for out, last_obj, _ in results:
        chunk_frames, missing = _fill_gaps(out, last_obj, source)
        frames.extend(chunk_frames)
        missing_total += missing
    return frames, missing_total


def decode_video_frames(
    source: VideoSource,
    *,
    num_frames: int | None = None,
    target_fps: float | None = None,
    sampling: Sampling = "uniform",
    start_frame: int | None = None,
    end_frame: int | None = None,
    min_frames: int = 1,
    max_frames: int | None = None,
    rounding: Rounding = "round",
    max_decoded_bytes: int | None = None,
    default_fps: float = _DEFAULT_FPS,
    threads: int | None = None,
) -> SampledVideo:
    """Decodes a subsampled set of frames from a video.

    Which frames to keep is decided before any pixels are produced, and the
    decode fetches only those, selecting each by its presentation timestamp so
    a damaged or variable-rate stream cannot silently yield the wrong frame.
    Peak memory is the sampled frames, bounded by ``max_decoded_bytes``.

    A container header that omits its frame count triggers a counting decode
    that retains no pixels; one that overcounts is corrected against the frames
    the decode actually produced.

    Args:
        source: The video to decode.
        num_frames: Sample this many frames. Mutually exclusive with
            ``target_fps``; when both are ``None`` every frame in range is
            kept, subject to the caps.
        target_fps: Sample the clip as if it ran at this frame rate. Never
            upsamples past the source rate.
        sampling: How ``target_fps`` selects frames -- ``"uniform"`` spreads
            the resulting count evenly, ``"walk"`` guarantees the spacing and
            the final frame.
        start_frame: First source frame eligible for sampling.
        end_frame: Last source frame eligible for sampling.
        min_frames: Never sample fewer than this many frames.
        max_frames: Model-specific ceiling on the sampled count.
        rounding: How to land fractional sample positions on real frame
            indices, for ``"uniform"`` sampling. Pick whichever matches the
            model's reference implementation.
        max_decoded_bytes: Ceiling on the decoded footprint of the sampled
            frames. The sampled count is reduced to fit.
        default_fps: Frame rate assumed when the container declares none.
        threads: Decode worker count; defaults to
            ``MODULAR_VIDEO_DECODE_THREADS`` or 16.

    Returns:
        The sampled frames, their source indices, and the stream metadata.

    Raises:
        InputError: If the source is not a decodable video, contains no
            decodable frames, or a single frame exceeds ``max_decoded_bytes``.
    """
    if num_frames is not None and target_fps is not None:
        raise ValueError("pass at most one of num_frames / target_fps")

    info = probe_video(source)
    fps = info.fps or default_fps
    budget_frames = _budget_frame_limit(info, max_decoded_bytes)
    worker_count = _decode_threads() if threads is None else max(1, threads)

    def plan(total: int) -> list[int]:
        return _plan_indices(
            total,
            fps,
            num_frames=num_frames,
            target_fps=target_fps,
            sampling=sampling,
            start_frame=start_frame,
            end_frame=end_frame,
            min_frames=min_frames,
            max_frames=max_frames,
            budget_frames=budget_frames,
            rounding=rounding,
        )

    with _wrap_decode_errors():
        total = info.num_frames
        if total <= 0:
            # The header offers no count; count with a decode that keeps no
            # pixels, so an unknown-length clip still costs O(1) memory.
            total = _count_frames(source)
        if total <= 0:
            raise MediaDecodeError("video contains no decodable frames")

        indices = plan(total)
        frames, missing = _decode_at_indices(source, indices, fps, worker_count)
        if missing:
            # The stream ended before some targets. Either the header
            # overcounted -- in which case the sample was spread over a
            # timeline that does not exist -- or the clip is truncated. Both
            # are answered by the true count, which is worth one pixel-free
            # counting pass rather than returning a sample padded with repeats.
            true_total = _count_frames(source)
            if true_total <= 0:
                raise MediaDecodeError("video contains no decodable frames")
            if true_total != total:
                total = true_total
                indices = plan(total)
                frames, missing = _decode_at_indices(
                    source, indices, fps, worker_count
                )
        if missing:
            logger.warning(
                "Video decode: %d of %d sampled frames were past the end of"
                " the decodable stream (damaged or truncated data);"
                " substituted the nearest decoded frame.",
                missing,
                len(indices),
            )

    return SampledVideo(
        frames=frames,
        indices=indices,
        info=replace(info, num_frames=total),
    )
