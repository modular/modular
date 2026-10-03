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
"""Captures the GPU kernels of the forward passes traced requests run in."""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
import queue
import re
import tempfile
import threading
import time
from bisect import bisect_left, bisect_right
from collections.abc import Mapping, Sequence
from typing import NamedTuple

from max._core.profiler import (
    load_profiler_plugin,
    start_capture,
    stop_capture,
)
from max.pipelines.context import TextContext
from max.pipelines.request import RequestID
from max.serve.config import KernelTraceLevel, Settings
from max.serve.telemetry import common as telemetry
from max.serve.telemetry._trace_context import (
    TRACE_LEVEL_HEADER,
    RequestTraceLevel,
)
from max.serve.telemetry.common import _span_exporter, logs_resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import (
    NonRecordingSpan,
    Span,
    SpanContext,
    Tracer,
    set_span_in_context,
)

_kernel_capture_permitted = False

# Bounds the passes one capture records, traced or not, since overlapping
# traced requests would otherwise keep it armed, and its buffers growing,
# indefinitely.
# TODO(MXTOOLS-651): The replay's 20,000-span cap and 64 MiB parse limit set
# this; raise it once the replay works in chunks.
_MAX_CAPTURE_PASSES = 64


def kernel_capture_permitted() -> bool:
    """Returns whether a request's ``x-max-trace-level`` header may arm a
    kernel capture in this model worker, as decided by
    :func:`configure_kernel_capture`."""
    return _kernel_capture_permitted


def _permit_kernel_capture(level: KernelTraceLevel) -> bool:
    """Loads the profiler plugin for per-request captures, or refuses them."""
    logger = logging.getLogger("max.serve")
    if level == KernelTraceLevel.KERNEL:
        # The session-wide capture holds the profiler for the whole run, so
        # no per-request capture could start.
        logger.warning(
            "Ignoring MAX_SERVE_KERNEL_TRACE_HEADERS: "
            "MAX_SERVE_KERNEL_TRACE_LEVEL=kernel already captures every pass."
        )
        return False
    if not load_profiler_plugin():
        logger.warning(
            "Ignoring MAX_SERVE_KERNEL_TRACE_HEADERS: no profiler is available."
        )
        return False
    return True


def configure_kernel_capture(settings: Settings) -> None:
    """Decides :func:`kernel_capture_permitted`, loading the profiler plugin
    when it is permitted.

    Must be called in the model worker after ``configure_tracing``, and
    before any device is created: on AMD, a plugin loaded later can't record.

    Args:
        settings: Server settings carrying ``kernel_trace_level``.
    """
    global _kernel_capture_permitted
    _kernel_capture_permitted = (
        telemetry._trace_level_header_enabled
        and _permit_kernel_capture(settings.kernel_trace_level)
    )


@dataclasses.dataclass(frozen=True)
class TracedPass:
    """A forward pass that ran under an armed kernel capture with at least one
    traced request."""

    batch_id: int
    level: RequestTraceLevel
    """The highest level among the pass's traced requests."""
    batch_span_context: SpanContext
    """The pass's ``max.batch`` span."""
    host_start_ns: int
    """``time.time_ns()`` taken before the pass's ``max.batch`` span started,
    a few microseconds before its ``max.batch`` range began."""
    host_end_ns: int
    """``time.time_ns()`` just after the range ended."""


_capture_dir: str | None = None


def kernel_capture_path() -> str:
    """Returns where this worker saves a per-request kernel capture: a fixed
    name in a private directory made on first use. Each capture overwrites
    the last, so the latest stays on disk for inspection."""
    global _capture_dir
    if _capture_dir is None:
        # The plugin's write follows symlinks, so a predictable name in a
        # shared temp directory would let another user aim it at any file
        # this worker can write.
        _capture_dir = tempfile.mkdtemp(prefix="max-kernel-capture-")
    return os.path.join(_capture_dir, "kernel-capture.json")


def _request_trace_level(context: TextContext) -> RequestTraceLevel | None:
    """Returns the level a request asked for with ``x-max-trace-level``."""
    if context.trace_carrier is None:
        return None
    value = context.trace_carrier.get(TRACE_LEVEL_HEADER)
    level = RequestTraceLevel.parse(value) if value else None
    return (
        level if level is not None and level > RequestTraceLevel.OFF else None
    )


class KernelCapture:
    """Captures the forward passes that traced requests run in.

    One capture is armed at the first pass with a traced request, shared by
    the traced requests that run while it records, and stopped when the last
    of them leaves or at the pass cap. The scheduler calls it from its own
    thread only.

    Args:
        max_passes: The most passes one capture records, traced or not.
    """

    def __init__(self, max_passes: int = _MAX_CAPTURE_PASSES) -> None:
        self._max_passes = max_passes
        self._traced: dict[RequestID, RequestTraceLevel] = {}
        self._traced_passes: list[TracedPass] = []
        self._passes = 0
        # Traced requests that ran in the armed capture.
        self._members: set[RequestID] = set()
        self._armed = False
        self._thread: KernelCaptureThread | None = None
        self._warned_not_started = False

    def admit(self, context: TextContext) -> bool:
        """Traces a new request if its ``x-max-trace-level`` header asks.

        Args:
            context: The request, as the scheduler admits it.

        Returns:
            Whether the request is traced.
        """
        level = _request_trace_level(context)
        if level is None:
            return False
        self._traced[context.request_id] = level
        return True

    def is_traced(self, request_id: RequestID) -> bool:
        """Returns whether a request is traced."""
        return request_id in self._traced

    @property
    def may_record(self) -> bool:
        """Whether a capture may be recording, so every pass needs a
        ``max.batch`` range for the replay to tell traced passes' GPU work
        from untraced ones'."""
        return self._armed or (self._thread is not None and self._thread.busy)

    def begin_pass(
        self, batches: Sequence[Sequence[TextContext]]
    ) -> RequestTraceLevel | None:
        """Arms the capture for a pass with traced members.

        Args:
            batches: The pass's batches, one per replica.

        Returns:
            The pass's level, or None when it has no traced member, the
            previous capture is still being saved, or another capture holds
            the profiler.
        """
        if not self._traced:
            return None
        members = [
            ctx.request_id
            for batch in batches
            for ctx in batch
            if ctx.request_id in self._traced
        ]
        if not members:
            return None
        if not self._armed:
            if self._thread is None:
                self._thread = KernelCaptureThread()
            elif self._thread.busy:
                return None
            if not start_capture():
                if not self._warned_not_started:
                    self._warned_not_started = True
                    logging.getLogger("max.serve").warning(
                        "Not capturing kernels for traced requests while "
                        "another capture or a Dynolog trace holds the "
                        "profiler."
                    )
                return None
            self._armed = True
        self._members.update(members)
        return max(self._traced[r] for r in members)

    def end_pass(
        self,
        batch_id: int,
        level: RequestTraceLevel | None,
        batch_span: Span,
        host_start_ns: int,
    ) -> list[RequestID]:
        """Records a pass that ran while the capture was armed, stopping the
        capture at the pass cap.

        Args:
            batch_id: The pass's batch ID.
            level: What :meth:`begin_pass` returned for the pass.
            batch_span: The pass's ``max.batch`` span.
            host_start_ns: ``time.time_ns()`` taken before the pass's
                ``max.batch`` span started, a few microseconds before its
                ``max.batch`` range began.

        Returns:
            The requests the pass cap untraced, which are still running.
        """
        if not self._armed:
            return []
        if level is not None:
            self._traced_passes.append(
                TracedPass(
                    batch_id=batch_id,
                    level=level,
                    batch_span_context=batch_span.get_span_context(),
                    host_start_ns=host_start_ns,
                    host_end_ns=time.time_ns(),
                )
            )
        self._passes += 1
        if self._passes >= self._max_passes:
            logging.getLogger("max.serve").warning(
                "Stopping a kernel capture at %d passes; its requests are no "
                "longer traced.",
                self._max_passes,
            )
            untraced = [
                r
                for r in self._members
                if self._traced.pop(r, None) is not None
            ]
            self._stop()
            return untraced
        return []

    def untrace(self, request_id: RequestID) -> None:
        """Untraces a request that has left, stopping the capture if it was
        the last traced one."""
        if (
            self._traced.pop(request_id, None) is None
            or self._traced
            or not self._armed
        ):
            return
        self._stop()

    def _stop(self) -> None:
        assert self._thread is not None
        self._armed = False
        self._passes = 0
        self._members.clear()
        passes, self._traced_passes = self._traced_passes, []
        self._thread.submit(kernel_capture_path(), passes)


class KernelCaptureThread:
    """Stops and saves per-request kernel captures off the scheduler thread.

    Saving holds the profiler's lock, so a start issued meanwhile would block
    the scheduler: :class:`KernelCapture` checks :attr:`busy` and defers
    re-arming instead. The replay also runs while busy, since the next
    capture overwrites its file.
    """

    def __init__(self) -> None:
        self._jobs: queue.SimpleQueue[tuple[str, list[TracedPass]]] = (
            queue.SimpleQueue()
        )
        self._idle = threading.Event()
        self._idle.set()
        threading.Thread(
            target=self._run, name="max-kernel-capture", daemon=True
        ).start()

    @property
    def busy(self) -> bool:
        """Whether a submitted capture is still being stopped, saved or
        replayed."""
        return not self._idle.is_set()

    def submit(self, output_path: str, passes: list[TracedPass]) -> None:
        """Queues a stop of the armed capture, saving it to ``output_path``."""
        self._idle.clear()
        self._jobs.put((output_path, passes))

    def _run(self) -> None:
        logger = logging.getLogger("max.serve")
        while True:
            output_path, passes = self._jobs.get()
            try:
                error = stop_capture(output_path)
                if error:
                    logger.warning("Kernel capture failed: %s", error)
                else:
                    logger.info(
                        "Saved a kernel capture of %d traced passes to %s",
                        len(passes),
                        output_path,
                    )
                    try:
                        replay_kernel_capture(output_path, passes)
                    except Exception:
                        logger.exception("Kernel capture replay failed")
            except Exception:
                logger.exception("Kernel capture failed")
            finally:
                self._idle.set()


_REPLAY_SPAN_CAP = 20_000
# Parsing holds the GIL, roughly 12 ms per MiB, so this bounds the stall to
# about 0.8 s; it peaks at about 6x the file size in memory.
# TODO(MXTOOLS-651): at about 1.2 KB per kernel, 64 MiB is about 55k kernels
# across all the worker's GPUs. A full 64-pass capture of an 8B-class model
# (~500 kernels a pass) fits, but a large MoE's (~2k a pass) does not; replay
# in chunks or parse off the GIL if those need spans.
_REPLAY_MAX_CAPTURE_BYTES = 64 * 1024 * 1024
# At ``kernel-sampled``, one pass in this many gets kernel spans.
_KERNEL_SAMPLE_STRIDE = 8
# libkineto kernel args that ``full`` adds as span attributes, each cast to
# one type since a capture can write occupancy 0 as an int.
_FULL_KERNEL_ARGS: tuple[tuple[str, str, type[int | float]], ...] = (
    ("registers per thread", "max.kernel.registers_per_thread", int),
    ("est. achieved occupancy %", "max.kernel.occupancy_pct", float),
    ("shared memory", "max.kernel.shared_mem", int),
)
_KINETO_GPU_CATEGORIES = frozenset({"kernel", "gpu_memcpy", "gpu_memset"})
_KINETO_LAUNCH_CATEGORIES = frozenset({"cuda_driver", "cuda_runtime"})
# libkineto stores range ids as 32-bit ints.
_KINETO_ID_MASK = 0xFFFF_FFFF
# Mojo kernel names end in a uniqueness hash, 8 hex digits, or 16 for a name
# shortened to 32 characters; it only adds noise to a span name.
_KERNEL_NAME_HASH = r"_[0-9a-f]{8}(?:[0-9a-f]{8})?$"
_replay_provider: TracerProvider | None = None


class _GpuActivity(NamedTuple):
    """A kernel, memcpy or memset, in libkineto's microseconds."""

    start_us: float
    end_us: float
    name: str
    args: Mapping[str, object]


# A span beneath ``max.batch.gpu``: name, start and end µs, and attributes.
_ChildSpan = tuple[str, float, float, dict[str, str | int | float]]


def _replay_tracer() -> Tracer:
    """Returns the replay's tracer, whose queue holds a whole capture.

    A separate provider keeps a capture's burst of spans from overflowing the
    worker's default 2,048-span queue and crowding out other requests' spans.
    """
    global _replay_provider
    if _replay_provider is None:
        _replay_provider = TracerProvider(resource=logs_resource)
        _replay_provider.add_span_processor(
            BatchSpanProcessor(
                _span_exporter(), max_queue_size=_REPLAY_SPAN_CAP
            )
        )
    return _replay_provider.get_tracer("max.serve.kernel_replay")


def _kineto_clock_offset_ns(
    ranges: Sequence[tuple[float, float, int]],
    passes: Mapping[int, TracedPass],
    base_ns: int,
) -> int:
    """Returns how far libkineto's clock runs ahead of ``time.time_ns()``.

    Each traced pass's host times bracket its ``max.batch`` range, bounding
    the offset. The bound's midpoint applies only when the bound excludes
    zero; otherwise the clocks are taken to agree.
    """
    lo, hi = -math.inf, math.inf
    for start_us, end_us, key in ranges:
        p = passes.get(key)
        if p is not None:
            lo = max(lo, base_ns + round(end_us * 1000) - p.host_end_ns)
            hi = min(hi, base_ns + round(start_us * 1000) - p.host_start_ns)
    if lo <= hi and (lo > 0 or hi < 0):
        return round((lo + hi) / 2)
    return 0


def _child_count(
    group: Sequence[_GpuActivity], level: RequestTraceLevel
) -> int:
    """Returns how many spans :func:`_child_spans` would build."""
    if level == RequestTraceLevel.OP:
        return len({a.name for a in group})
    if level in (RequestTraceLevel.KERNEL, RequestTraceLevel.FULL):
        return len(group)
    return 0


def _child_spans(
    batch_id: int, group: Sequence[_GpuActivity], level: RequestTraceLevel
) -> list[_ChildSpan]:
    """Returns the spans under a pass's ``max.batch.gpu`` span at ``level``."""
    if level == RequestTraceLevel.OP:
        by_name: dict[str, list[_GpuActivity]] = {}
        for a in group:
            by_name.setdefault(a.name, []).append(a)
        aggregates: list[_ChildSpan] = []
        for name, same in by_name.items():
            # Rounded to libkineto's ns resolution, since differences of
            # large µs offsets carry float noise; exact below 2^43 µs.
            durations = [round(a.end_us - a.start_us, 3) for a in same]
            total = round(sum(durations), 3)
            aggregates.append(
                (
                    "kernel_agg:" + re.sub(_KERNEL_NAME_HASH, "", name),
                    min(a.start_us for a in same),
                    max(a.end_us for a in same),
                    {
                        "max.batch_id": batch_id,
                        "max.kernel.name": name,
                        "max.kernel.count": len(same),
                        "max.kernel.total_us": total,
                        "max.kernel.mean_us": round(total / len(same), 3),
                        "max.kernel.max_us": max(durations),
                    },
                )
            )
        return aggregates
    if level not in (RequestTraceLevel.KERNEL, RequestTraceLevel.FULL):
        return []
    kernels: list[_ChildSpan] = []
    for a in group:
        attrs: dict[str, str | int | float] = {
            "max.batch_id": batch_id,
            "max.kernel.name": a.name,
        }
        if level == RequestTraceLevel.FULL:
            for arg, attr, kind in _FULL_KERNEL_ARGS:
                value = a.args.get(arg)
                if isinstance(value, (int, float)) and not isinstance(
                    value, bool
                ):
                    attrs[attr] = kind(value)
        kernels.append(
            (re.sub(_KERNEL_NAME_HASH, "", a.name), a.start_us, a.end_us, attrs)
        )
    return kernels


def replay_kernel_capture(
    output_path: str, passes: Sequence[TracedPass]
) -> None:
    """Exports a stopped capture's GPU activity as spans of its traced passes.

    An activity belongs to the pass whose ``max.batch`` range last started
    at or before its launch record, joined on ``correlation``. Launch time
    decides because under overlap scheduling a pass's GPU work trails its
    host range. An activity with no launch record, such as a copy or a
    device graph's kernel, takes the pass of the launch records either side
    of it in ``correlation`` order. Where those lie in different passes, its
    ``External id`` must name one of the passes between them, or it is
    dropped. Each traced pass with captured activity gets a backdated
    ``max.batch.gpu`` span under its ``max.batch`` span. Beneath that,
    ``op`` adds one aggregate span per kernel name, ``kernel`` a span per
    activity, ``kernel-sampled`` the same on one pass in eight, and ``full``
    adds registers, occupancy and shared memory to kernel spans. Activities
    of untraced passes are dropped.

    Args:
        output_path: The libkineto Chrome-trace JSON the capture wrote.
        passes: The capture's traced passes.
    """
    # A pass's replayed spans share its max.batch span's trace and sampler,
    # so they'd be dropped wherever it was; parsing for them would hold the
    # GIL for nothing.
    passes = [p for p in passes if p.batch_span_context.trace_flags.sampled]
    if not passes:
        return
    log = logging.getLogger("max.serve")
    ranges: list[tuple[float, float, int]] = []
    launch_us: dict[int, float] = {}
    activities: list[tuple[_GpuActivity, int, int]] = []
    try:
        size = os.path.getsize(output_path)
        if size > _REPLAY_MAX_CAPTURE_BYTES:
            log.warning(
                "Kernel capture %s is %d bytes, over the %d-byte replay"
                " limit; not exporting its spans",
                output_path,
                size,
                _REPLAY_MAX_CAPTURE_BYTES,
            )
            return
        with open(output_path, encoding="utf-8") as f:
            doc = json.load(f)
        base_ns = int(doc.get("baseTimeNanoseconds", 0))
        for e in doc["traceEvents"]:
            if e.get("ph") != "X":
                continue
            cat = e.get("cat")
            args = e.get("args") or {}
            if cat in _KINETO_GPU_CATEGORIES:
                ts = float(e["ts"])
                activity = _GpuActivity(
                    ts,
                    ts + float(e.get("dur", 0)),
                    str(e.get("name", "")),
                    args,
                )
                activities.append(
                    (
                        activity,
                        int(args.get("correlation", -1)),
                        int(args.get("External id", 0)) & _KINETO_ID_MASK,
                    )
                )
            elif cat in _KINETO_LAUNCH_CATEGORIES and "correlation" in args:
                launch_us[int(args["correlation"])] = float(e["ts"])
            elif cat == "user_annotation" and e.get("name") == "max.batch":
                ts = float(e["ts"])
                # libkineto omits an External id of 0.
                ranges.append(
                    (
                        ts,
                        ts + float(e.get("dur", 0)),
                        int(args.get("External id", 0)) & _KINETO_ID_MASK,
                    )
                )
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as err:
        log.warning("Cannot read kernel capture %s: %s", output_path, err)
        return

    by_key = {p.batch_id & _KINETO_ID_MASK: p for p in passes}
    ranges.sort()
    range_starts = [r[0] for r in ranges]
    range_index = {key: k for k, (_, _, key) in enumerate(ranges)}
    launches = sorted(launch_us.items())
    launch_correlations = [c for c, _ in launches]
    cut_off = len(ranges)

    def pass_at(launched_us: float) -> int:
        """Returns the index of the last range to start at or before a
        launch, -1 if none did, or ``cut_off`` for a launch after the last
        range's end, which belongs to a pass whose range the stop cut off."""
        i = bisect_right(range_starts, launched_us) - 1
        if i >= 0 and i + 1 == cut_off and launched_us > ranges[i][1]:
            return cut_off
        return i

    grouped: dict[int, list[_GpuActivity]] = {}
    for activity, correlation, external_id in activities:
        launched = launch_us.get(correlation)
        key: int | None = None
        if launched is not None:
            i = pass_at(launched)
            if 0 <= i < cut_off:
                key = ranges[i][2]
        else:
            # libkineto records kernel launches only, not graph launches or
            # copies. With no launch record after it, the activity may be
            # the cut-off pass's.
            j = bisect_left(launch_correlations, correlation)
            lo = pass_at(launches[j - 1][1]) if j > 0 else -1
            hi = pass_at(launches[j][1]) if j < len(launches) else cut_off
            lo, hi = min(lo, hi), max(lo, hi)
            if lo == hi:
                key = ranges[lo][2] if 0 <= lo < cut_off else None
            elif max(lo, 0) <= range_index.get(external_id, -1) <= hi:
                # Mojo op ranges reuse small External ids, so it is trusted
                # only to pick among the passes the launches bracket.
                key = external_id
        if key is not None and key in by_key:
            grouped.setdefault(key, []).append(activity)

    if not grouped:
        return
    offset_ns = _kineto_clock_offset_ns(ranges, by_key, base_ns)

    def to_ns(us: float) -> int:
        return base_ns + round(us * 1000) - offset_ns

    tracer = _replay_tracer()
    keys = sorted(grouped, key=lambda k: by_key[k].batch_id)
    # Reserves the cap for every pass's max.batch.gpu span before any span
    # beneath one.
    child_budget = _REPLAY_SPAN_CAP - min(len(keys), _REPLAY_SPAN_CAP)
    dropped = 0
    sampled_seen = 0
    for n, key in enumerate(keys):
        p = by_key[key]
        group = grouped[key]
        level = p.level
        if level == RequestTraceLevel.KERNEL_SAMPLED:
            sampled = sampled_seen % _KERNEL_SAMPLE_STRIDE == 0
            level = RequestTraceLevel.KERNEL if sampled else level
            sampled_seen += 1
        if n >= _REPLAY_SPAN_CAP:
            dropped += 1 + _child_count(group, level)
            continue
        gpu_span = tracer.start_span(
            "max.batch.gpu",
            context=set_span_in_context(NonRecordingSpan(p.batch_span_context)),
            start_time=to_ns(min(a.start_us for a in group)),
            attributes={"max.batch_id": p.batch_id},
        )
        children: list[_ChildSpan] = []
        if child_budget == 0:
            dropped += _child_count(group, level)
        else:
            children = _child_spans(p.batch_id, group, level)
        child_ctx = set_span_in_context(gpu_span)
        for name, start_us, end_us, attrs in children:
            if child_budget == 0:
                dropped += 1
                continue
            child_budget -= 1
            span = tracer.start_span(
                name,
                context=child_ctx,
                start_time=to_ns(start_us),
                attributes=attrs,
            )
            span.end(end_time=to_ns(end_us))
        gpu_span.end(end_time=to_ns(max(a.end_us for a in group)))
    if dropped:
        log.warning(
            "Kernel capture replay hit its %d-span cap; dropped %d spans",
            _REPLAY_SPAN_CAP,
            dropped,
        )
