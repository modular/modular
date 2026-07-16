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

"""State-leak gate for the disabled libkineto profiler hot path (MXTOOLS-190).

When the profiler is disabled, a single relaxed atomic load (``isEnabled()``)
gates every ``rangeBegin`` / ``rangeEnd`` / ``step()`` call. That gate is always
compiled in, so this test does *not* measure its per-call cost — the ≤0.2%
off-cost is a design property of that single relaxed load (see Range.h); no
current test times it per-call. ``//Support/unittests/Profiling:RangeTest``
pins the functional contracts around it (enable/disable idempotence, fork
safety, scope destruction) without any timing assertions.

What this test adds is the one thing a unit test of the gate can't see: that an
enable/disable round-trip leaves *no perf-visible residue* in the disabled
kernel-launch path. It times a tight loop of disabled CUDA kernel launches,
does a no-op enable/disable round-trip, then times the same loop again. A
materially slower second run means state from the cycle leaked into the
disabled path — e.g. a flag ``disable()`` failed to clear, or a CUPTI
subscription that survives ``disable()`` and keeps detouring every
``cuLaunchKernel`` below the ``isEnabled()`` gate (a cost inside the driver's
launch path that only a test launching real kernels in the off state can see).

This catches regressions *introduced by* an enable/disable cycle, not a static
unconditional addition to the hot path — that would appear in both runs equally
and produce zero drift. Absolute off-cost is not asserted by any test today;
it is bounded by construction to the single relaxed load.

A tiny two-input add graph is used rather than a real LLM: it isolates
per-kernel-launch cost, runs in seconds with no weights / network / tokenizer
dependencies, and avoids LLM-level latency variance that would dwarf the
disabled-path delta we are trying to detect.

Thresholds
==========

- ``MAX_PROFILER_DEACTIVATED_DRIFT_LIMIT`` (default 25%) — the second disabled run's
  best-case (minimum) per-iter time must be within this fraction of the
  first run's best-case.  We don't bind it to the 0.2% marketing number
  because end-to-end timing variance on real hardware (CUDA driver kernel
  scheduling, host clock jitter, page-table churn, co-tenant contention on
  shared GPU runners) dwarfs it: measured best-case drift on a ~25µs op on
  a B200 ran 0.6-12.5% across most no-change runs, with one outlier at
  37.9% when a co-tenant burst covered an entire (then ~5ms) sample batch.
  The outlier is addressed by widening the batch (see ``_TIMED_ITERS``)
  rather than loosening the gate; 25% (~2x the worst non-outlier) still
  catches an order-of-magnitude regression.  The 0.2% applies to *per-call*
  overhead, far below what end-to-end iteration timing can resolve.
- ``MAX_PROFILER_ACTIVATED_OVERHEAD_LIMIT`` (default 25%) — when the profiler is enabled,
  inference may be measurably slower (libkineto's CUPTI hooks fire on
  every kernel launch). We pick 25% as a sanity ceiling: real enabled
  overhead on a CUDA workload is typically 5-15%, but the gate's job is
  to catch catastrophic regressions (enabled run 2x slower), not to track
  small fluctuations.
"""

import os
import time
from collections.abc import Iterator

import numpy as np
import pytest
from max._core.profiler import (
    kineto_can_record,
    kineto_disable,
    kineto_enable,
    kineto_have_libkineto,
    kineto_is_enabled,
)
from max.driver import Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, ops

# Tight loop size; large enough that the per-batch minimum settles onto the
# achievable floor but small enough that the timed loops stay a small fraction
# of the test's wall time (session setup dominates at ~20s).  Tunable from the
# env so CI can dial it up for a slower runner without recompiling.
#
# Default 2000: the minimum of a batch only lands on the true floor if the
# batch spans enough wall-clock that a transient co-tenant burst on a shared
# GPU runner cannot cover all of it.  At 200 iterations a batch of ~25µs ops
# lasts ~5ms, and a single few-ms preemption swallowed an entire batch on the
# shared b200 pool, producing a 37.9% no-change "drift" failure (~1-in-12
# flake rate).  At 2000 iterations a batch spans ~50ms, so the minimum
# reliably includes un-preempted iterations.
_WARMUP_ITERS = int(os.environ.get("MAX_PROFILER_WARMUP_ITERS", "5"))
_TIMED_ITERS = int(os.environ.get("MAX_PROFILER_TIMED_ITERS", "2000"))

# Drift gates — see module docstring for rationale.  Env-overridable so CI
# can relax (or tighten) without a code change.  Named ``..._LIMIT`` (not a bare
# ``MAX_`` prefix) so the ceiling reads as "maximum" and not the MAX product.
_DEACTIVATED_DRIFT_LIMIT = float(
    os.environ.get("MAX_PROFILER_DEACTIVATED_DRIFT_LIMIT", "0.25")
)
_ACTIVATED_OVERHEAD_LIMIT = float(
    os.environ.get("MAX_PROFILER_ACTIVATED_OVERHEAD_LIMIT", "0.25")
)

_skip_without_accelerator = pytest.mark.skipif(
    not accelerator_count(),
    reason="Profiler overhead gate requires a real GPU kernel-launch loop.",
)


@pytest.fixture(autouse=True)
def _disable_profiler_after_each_test() -> Iterator[None]:
    yield
    if kineto_is_enabled():
        kineto_disable()


def _build_tiny_add_graph() -> Graph:
    """Smallest graph that exercises a real CUDA kernel launch.

    Two-input add on GPU(0); ``model.execute(a, b)`` then triggers exactly
    the kernel-launch path libkineto's CUPTI callbacks watch.
    """
    input_type = TensorType(
        dtype=DType.float32,
        shape=["batch", "channels"],
        device=DeviceRef.GPU(0),
    )
    with Graph(
        "profiler_overhead_add", input_types=(input_type, input_type)
    ) as graph:
        out = ops.add(graph.inputs[0], graph.inputs[1])
        graph.output(out)
    return graph


@pytest.fixture
def warm_model() -> tuple[Model, Buffer, Buffer]:
    """Compiled tiny-add model plus its two device inputs, post-warmup.

    Shared by both overhead tests: builds the accelerator session, allocates
    the two device-resident inputs the graph declares (``DeviceRef.GPU(0)``),
    and runs the warmup loop so the timed loops measure steady-state
    kernel-launch cost rather than first-call compilation / module-load cost.
    Allocated once and reused across iterations so the timed loop measures the
    kernel-launch path, not per-call host->device transfers.
    """
    device = Accelerator()
    session = InferenceSession(devices=[device])
    model = session.load(_build_tiny_add_graph())

    a = Buffer.from_numpy(np.ones((4, 8), dtype=np.float32)).to(device)
    b = Buffer.from_numpy(np.full((4, 8), 2.0, dtype=np.float32)).to(device)

    # Warmup: page in kernels, populate CUDA module cache, settle clocks.
    for _ in range(_WARMUP_ITERS):
        model.execute(a, b)

    return model, a, b


def _min_iter_seconds(model: Model, a: Buffer, b: Buffer, iters: int) -> float:
    """Best-case (minimum) wall-clock time per iteration over ``iters`` calls.

    Minimum, not median: for a sub-millisecond micro-op the per-call cost has
    a hard floor (the real kernel-launch + dispatch cost) and a long upward
    tail from noise the test does not care about — GC pauses, CUDA driver
    scheduler jitter, host clock interrupts, contention from other tenants on a
    shared GPU runner.  The median still tracks that tail, so on a ~27µs op the
    median of two 200-sample batches swings run-to-run by tens of percent even
    with no code change, which makes a tight drift gate flaky.  The minimum
    filters the tail down to the achievable floor, which is stable across runs
    and is the quantity a "did the disabled hot path get slower" gate actually
    wants to compare.  ``time.perf_counter`` is the highest-resolution monotonic
    clock Python exposes and is the right call for these intervals.
    """
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        model.execute(a, b)
        samples.append(time.perf_counter() - start)
    return min(samples)


@_skip_without_accelerator
def test_disabled_overhead_is_stable_across_enable_cycle(
    warm_model: tuple[Model, Buffer, Buffer],
) -> None:
    """A no-op enable/disable round-trip must not move the disabled hot path.

    Method: time N kernel launches with profiler disabled (the steady-state
    user experience), do an enable/disable round-trip without ever recording,
    time N more disabled launches, assert the best-case delta stays within
    ``_DEACTIVATED_DRIFT_LIMIT``.

    What this catches — residue an enable/disable cycle leaves in the disabled
    path:

    - A pending state-machine flag that ``disable()`` failed to clear and
      that ``isEnabled()`` consults on the hot path.
    - A leaked CUPTI subscription that survives ``disable()`` and slows
      kernel launches.
    - Any syscall / lock / non-relaxed atomic an enable/disable cycle leaves
      armed on ``rangeBegin`` / ``step()``.  (A static, unconditional addition
      is present in both runs, nets zero drift, and is out of scope — see the
      module docstring.)
    """
    model, a, b = warm_model

    assert not kineto_is_enabled(), (
        "expected disabled steady state before measurement"
    )
    baseline = _min_iter_seconds(model, a, b, _TIMED_ITERS)

    # Round-trip enable/disable without ever calling start()/stop() on a real
    # capture window.  We run this unconditionally — not gated on
    # ``kineto_can_record()`` — because ``enable()``/``disable()`` flip the
    # enabled flag that ``isEnabled()`` reads on the hot path (the
    # ``getKinetoEnabled()`` local-static, plus the ``getProfilerState()``
    # observability state behind ``state()``) regardless of whether a CUDA
    # context is live; only the CUPTI ``startTrace`` / ``stopTrace`` is gated
    # on a live context.  Gating the round-trip on recording made this test
    # silently inert when no context was bound on the calling thread, which is
    # exactly the leaked-flag scenario the test is meant to catch.
    kineto_enable()
    kineto_disable()
    assert not kineto_is_enabled()

    second = _min_iter_seconds(model, a, b, _TIMED_ITERS)

    # One-sided gate: only a *slower* post-cycle run is a regression.  A faster
    # second run (warmer caches / clocks) yields negative drift and must pass —
    # a two-sided ``abs()`` would turn that good outcome into a red build.
    drift = (second - baseline) / baseline
    assert drift < _DEACTIVATED_DRIFT_LIMIT, (
        f"disabled hot-path drift {drift:.2%} exceeds gate "
        f"{_DEACTIVATED_DRIFT_LIMIT:.0%}: baseline best-case {baseline * 1e6:.1f}µs vs "
        f"post-cycle best-case {second * 1e6:.1f}µs (over {_TIMED_ITERS} iters "
        f"each).  Likely cause: a new operation on the disabled hot path or a "
        f"state leak from enable()/disable()."
    )


@_skip_without_accelerator
def test_enabled_overhead_within_sanity_ceiling(
    warm_model: tuple[Model, Buffer, Buffer],
) -> None:
    """Enabled-mode overhead must stay within a sanity ceiling vs disabled.

    This is a coarse catastrophic-regression gate, not a sub-percent
    tracker.  Real enabled-mode overhead on a CUDA workload is typically
    5-15% (libkineto's CUPTI hooks fire on every kernel launch and push
    correlation IDs).  If a change pushes enabled overhead past
    ``_ACTIVATED_OVERHEAD_LIMIT`` we want to know — but we deliberately don't
    pin it tighter because absolute enabled overhead depends on driver
    version, kernel size, and CUPTI configuration in ways that fluctuate
    independently of MAX-side code.
    """
    model, a, b = warm_model

    # ``kineto_can_record()`` requires the libkineto backend, which is linked
    # only in ``--config=kineto`` builds on Linux x86_64 (#91288) — default
    # builds self-skip here.  Probed at runtime (not in a module-level
    # ``skipif``) so the skip reflects the state after the fixture has built
    # a session.  Recording additionally requires a CUDA primary context
    # current on *this* thread (the probe is ``cuCtxGetCurrent`` via the
    # on-demand stub in ``bazel/third-party/libcuda_stub.cpp``), so the test
    # can also skip on kineto builds if the session's context is not bound
    # on the pytest thread.
    if not kineto_can_record():
        reason = (
            "the libkineto backend is not linked (requires --config=kineto)"
            if not kineto_have_libkineto()
            else "no CUDA primary context is current on the test thread"
        )
        pytest.skip(f"enabled-mode overhead is unmeasurable here: {reason}.")

    disabled = _min_iter_seconds(model, a, b, _TIMED_ITERS)

    kineto_enable()
    try:
        enabled = _min_iter_seconds(model, a, b, _TIMED_ITERS)
    finally:
        kineto_disable()

    overhead = (enabled - disabled) / disabled
    assert overhead < _ACTIVATED_OVERHEAD_LIMIT, (
        f"enabled-mode overhead {overhead:.2%} exceeds ceiling "
        f"{_ACTIVATED_OVERHEAD_LIMIT:.0%}: disabled best-case {disabled * 1e6:.1f}µs vs "
        f"enabled best-case {enabled * 1e6:.1f}µs (over {_TIMED_ITERS} iters each).  "
        f"Likely cause: a new per-kernel-launch CUPTI callback or a wider "
        f"libkineto activity set."
    )
