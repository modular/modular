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

"""Integration tests for the ``session.profiling`` control surface (MXTOOLS-190).

Covers the public Python surface end-to-end: state-machine transitions,
``session.debug.profiling_output_path`` round-tripping, output-path template
expansion ({pid}/{rank}/directory mode), and the ``ProfilingError`` raised by
``wait_for_trace()`` on serialization failure.

Tests that require an actual libkineto-written trace file (the path-expansion
and error-path cases) skip automatically when ``kineto_can_record()`` is
false — libkineto not compiled into the profiler core, or no CUDA primary
context bound. Today that means they skip in every configuration (the
recording path is not yet wired through the libkineto backend), so the file
runs cleanly under the same ``./bazelw test`` invocation on every developer
machine; they arm automatically once recording lands.
"""

import json
import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from max._core.profiler import (
    kineto_can_record,
    kineto_disable,
    kineto_is_enabled,
)
from max.driver import CPU
from max.engine import InferenceSession, ProfilingError

# Tests that assert on the produced trace file (or the absence/contents of
# ``lastTraceError``) need ``disable()`` to actually exercise the libkineto
# write path. That path is gated on both the build (``MODULAR_HAVE_KINETO``)
# and a live CUDA primary context at enable() time, so we skip when either is
# missing. Today ``kineto_can_record()`` is false in EVERY configuration —
# the recording path is not yet wired through the libkineto backend (see
# haveLibkineto() in Support/Profiling/Range.h) — so the gated tests below
# skip everywhere, including GPU Linux x86_64 hosts, and start running once
# the wiring lands.
#
# The probe itself is side-effect-free: ``kineto_can_record()`` boils down to
# ``cuCtxGetCurrent`` on the calling thread, and deliberately does NOT
# manufacture a context (no cuInit / cuDevicePrimaryCtxRetain — see
# RangeKineto.cpp). Because this decorator is evaluated at module import /
# collection time, the skip decision is frozen before any session exists:
# once recording is wired up, running the gated tests will require a CUDA
# primary context to already be bound on the collecting thread at import
# time (e.g. by a conftest fixture), or this gate to move inside the tests.
_skip_without_recording = pytest.mark.skipif(
    not kineto_can_record(),
    reason=(
        "libkineto recording path inactive — libkineto is not compiled into "
        "the profiler core (true of every current build), or no CUDA primary "
        "context is bound on this host."
    ),
)


@pytest.fixture(autouse=True)
def _disable_profiler_after_each_test() -> Iterator[None]:
    """Restore the process-wide profiler to disabled after every test.

    The libkineto profiler is a process-global singleton, so a test that
    enables it then fails before reaching its own ``stop()`` call would leave
    every subsequent test starting from an enabled state. This fixture runs
    teardown regardless of test outcome.
    """
    yield
    if kineto_is_enabled():
        kineto_disable()


@pytest.fixture
def _reset_profiling_output_path() -> Iterator[InferenceSession]:
    """Snapshot and restore ``debug.profiling_output_path`` around a test.

    The DebugConfig is a process-wide singleton, so a test that sets the
    path would otherwise leak into every later test in the same pytest
    invocation. Restoring it explicitly keeps the suite hermetic.

    Yields the session it created so tests can reuse it rather than
    constructing a second one — ``session.debug`` is process-global, so the
    instance is incidental anyway.
    """
    session = _new_session()
    original = session.debug.profiling_output_path
    try:
        yield session
    finally:
        session.debug.profiling_output_path = original


@pytest.fixture
def _scrubbed_rank_env() -> Iterator[None]:
    """Unset rank env vars for the duration of one test.

    ``expandOutputPath`` reads ``MODULAR_RANK`` first, then
    ``OMPI_COMM_WORLD_RANK``. A stray value inherited from the surrounding
    shell or a sibling test would shadow whatever the test itself sets, so
    we scrub both and restore on teardown.
    """
    saved = {
        k: os.environ.pop(k, None)
        for k in ("MODULAR_RANK", "OMPI_COMM_WORLD_RANK")
    }
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _new_session() -> InferenceSession:
    return InferenceSession(devices=[CPU()])


def _capture_to(session: InferenceSession, path: str) -> None:
    """Run one full capture cycle writing to ``path``.

    Single-sources the set-path / start / stop / wait_for_trace ceremony the
    file-creation tests share. The error-path test deliberately keeps these
    steps split (it wraps ``wait_for_trace()`` in ``pytest.raises``), so it
    does not use this helper.
    """
    session.debug.profiling_output_path = path
    session.profiling.start()
    session.profiling.stop()
    session.profiling.wait_for_trace()


def test_profiling_namespace_exists() -> None:
    session = _new_session()
    assert hasattr(session, "profiling")
    assert hasattr(session.profiling, "start")
    assert hasattr(session.profiling, "stop")
    assert hasattr(session.profiling, "wait_for_trace")
    assert hasattr(session.profiling, "state")
    assert hasattr(session.profiling, "is_enabled")


def test_disabled_by_default() -> None:
    session = _new_session()
    assert session.profiling.state == "idle"
    assert session.profiling.is_enabled is False


def test_profiling_namespace_is_shared_across_sessions() -> None:
    # libkineto's profiler state is process-global, so ``session.profiling``
    # must be the same object on every ``InferenceSession`` — a regression
    # that moved the namespace into ``__init__`` (per-instance) would silently
    # flip the contract and break multi-session orchestration.
    s1 = _new_session()
    s2 = _new_session()
    assert s1.profiling is s2.profiling
    s1.profiling.start()
    assert s2.profiling.is_enabled is True
    assert s2.profiling.state == "warmup"


def test_start_transitions_to_warmup() -> None:
    # enable() lands in "warmup", not "active": the Warmup -> Active
    # step-count transition is not wired yet (see step()'s TODO, MXTOOLS-190).
    # When the warmup/active step machine lands, start() will report "warmup"
    # until the configured warmup steps elapse and then advance to "active".
    #
    # The autouse ``_disable_profiler_after_each_test`` fixture restores the
    # profiler to disabled on teardown, so no in-test ``finally`` is needed.
    session = _new_session()
    session.profiling.start()
    assert session.profiling.state == "warmup"
    assert session.profiling.is_enabled is True


def test_stop_returns_to_idle() -> None:
    session = _new_session()
    session.profiling.start()
    session.profiling.stop()
    assert session.profiling.state == "idle"
    assert session.profiling.is_enabled is False


def test_double_start_is_idempotent() -> None:
    # Calling start() twice in a row must leave the observable state
    # unchanged — the second call is a no-op.
    session = _new_session()
    session.profiling.start()
    state_after_first = session.profiling.state
    assert state_after_first == "warmup"
    assert session.profiling.is_enabled is True

    session.profiling.start()
    assert session.profiling.state == state_after_first
    assert session.profiling.is_enabled is True


def test_double_stop_is_idempotent() -> None:
    # Calling stop() twice in a row must not raise and must leave the
    # profiler in "idle" — assert on the resolved state, not just
    # is_enabled, so a regression that lands in "flushing" is caught.
    session = _new_session()
    session.profiling.start()
    session.profiling.stop()
    assert session.profiling.state == "idle"
    assert session.profiling.is_enabled is False

    session.profiling.stop()
    assert session.profiling.state == "idle"
    assert session.profiling.is_enabled is False


def test_wait_for_trace_after_stop() -> None:
    # In the current skeleton, wait_for_trace() is synchronous (no libkineto
    # serialization thread yet), so this just verifies the entry point is
    # callable. The asynchronous wait gets coverage once the
    # libkineto integration lands.
    session = _new_session()
    session.profiling.start()
    session.profiling.stop()
    session.profiling.wait_for_trace()


def test_context_manager_starts_and_stops() -> None:
    session = _new_session()
    assert session.profiling.is_enabled is False
    with session.profiling:
        assert session.profiling.is_enabled is True
        assert session.profiling.state == "warmup"
    assert session.profiling.is_enabled is False
    assert session.profiling.state == "idle"


def test_context_manager_stops_on_exception() -> None:
    # __exit__ must call stop() even when the body raises, so the
    # process-global profiler doesn't leak into subsequent code.
    session = _new_session()

    class _Sentinel(Exception):
        pass

    with pytest.raises(_Sentinel):
        with session.profiling:
            assert session.profiling.is_enabled is True
            raise _Sentinel
    assert session.profiling.is_enabled is False
    assert session.profiling.state == "idle"


# ---------------------------------------------------------------------------
# profiling_output_path round-trip — DebugConfig setter / getter parity.
# ---------------------------------------------------------------------------


def test_profiling_output_path_roundtrips(
    tmp_path: Path, _reset_profiling_output_path: InferenceSession
) -> None:
    session = _reset_profiling_output_path
    explicit = str(tmp_path / "explicit-trace.json")
    session.debug.profiling_output_path = explicit
    assert session.debug.profiling_output_path == explicit


def test_profiling_output_path_accepts_empty(
    _reset_profiling_output_path: InferenceSession,
) -> None:
    # Empty string means "fall back to Range.cpp's built-in default".  Round-
    # tripping it through the setter must not coerce it into None or raise —
    # users hit this path via ``unset`` or by clearing the env var.
    session = _reset_profiling_output_path
    session.debug.profiling_output_path = ""
    assert session.debug.profiling_output_path == ""


# ---------------------------------------------------------------------------
# Output-path expansion — end-to-end through Python.  Requires libkineto so
# disable() actually serializes a file.
# ---------------------------------------------------------------------------


@_skip_without_recording
def test_pid_template_expansion_writes_to_expanded_path(
    tmp_path: Path, _reset_profiling_output_path: InferenceSession
) -> None:
    template = str(tmp_path / "trace-{pid}.json")
    expected = tmp_path / f"trace-{os.getpid()}.json"

    _capture_to(_reset_profiling_output_path, template)

    assert expected.exists(), (
        f"libkineto did not write the expanded path {expected!s}; "
        f"directory listing: {sorted(p.name for p in tmp_path.iterdir())}"
    )
    # Trace files are Chrome-trace JSON; a successful write yields at least
    # an opening "{".  We don't assert on full schema here — that's HTA's job
    # in test_hta_import once GPU runners are available (MXTOOLS-190).
    assert expected.read_bytes().lstrip().startswith(b"{")


@_skip_without_recording
def test_rank_template_resolves_modular_rank(
    tmp_path: Path,
    _reset_profiling_output_path: InferenceSession,
    _scrubbed_rank_env: None,
) -> None:
    # MODULAR_RANK must win even when OMPI_COMM_WORLD_RANK is also set — set
    # both and assert the file resolves to MODULAR_RANK's value, pinning the
    # precedence in readRankEnv() directly (not just "OMPI is read when
    # MODULAR is absent", which the fallback test below already covers).
    os.environ["MODULAR_RANK"] = "4"
    os.environ["OMPI_COMM_WORLD_RANK"] = "9"
    template = str(tmp_path / "trace-{rank}.json")
    expected = tmp_path / "trace-4.json"

    _capture_to(_reset_profiling_output_path, template)

    assert expected.exists()


@_skip_without_recording
def test_rank_template_falls_back_to_ompi(
    tmp_path: Path,
    _reset_profiling_output_path: InferenceSession,
    _scrubbed_rank_env: None,
) -> None:
    # OMPI_COMM_WORLD_RANK is the MPI-launcher fallback; expandOutputPath
    # consults it only when MODULAR_RANK is unset.  Asserting the precedence
    # here keeps the contract honest as multi-rank captures grow.
    os.environ["OMPI_COMM_WORLD_RANK"] = "7"
    template = str(tmp_path / "trace-{rank}.json")
    expected = tmp_path / "trace-7.json"

    _capture_to(_reset_profiling_output_path, template)

    assert expected.exists()


@_skip_without_recording
def test_directory_output_mode_generates_per_pid_file(
    tmp_path: Path,
    _reset_profiling_output_path: InferenceSession,
    _scrubbed_rank_env: None,
) -> None:
    # When the configured path resolves to an existing directory, disable()
    # writes "trace_rank<rank>_<pid>_<unix-ts>_<seq>.json" inside it — see
    # Detail::expandOutputPath.  The rank-env fixture scrubs MODULAR_RANK /
    # OMPI_COMM_WORLD_RANK so expandOutputPath falls back to "0".  The
    # timestamp and per-process sequence portions are non-deterministic, so
    # we match on shape (prefix + suffix) only.
    _capture_to(_reset_profiling_output_path, str(tmp_path))

    prefix = f"trace_rank0_{os.getpid()}_"
    matches = [
        p
        for p in tmp_path.iterdir()
        if p.name.startswith(prefix) and p.name.endswith(".json")
    ]
    assert len(matches) == 1, (
        f"expected exactly one {prefix}*.json under {tmp_path}, "
        f"found: {sorted(p.name for p in tmp_path.iterdir())}"
    )


# ---------------------------------------------------------------------------
# Error reporting — wait_for_trace() must raise ProfilingError if libkineto
# could not write the trace.
# ---------------------------------------------------------------------------


@_skip_without_recording
def test_wait_for_trace_raises_on_unwritable_path(
    tmp_path: Path, _reset_profiling_output_path: InferenceSession
) -> None:
    # Range.cpp's disable() mkdir -p's the parent before saving, so a merely
    # missing parent directory would be created rather than failing.  To force
    # an unwritable path we plant a regular *file* where a directory would need
    # to be: create_directories() then fails and libkineto's save() cannot open
    # the target, so disable() records a ProfilingError that wait_for_trace()
    # surfaces as a Python exception.
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    bad = blocker / "trace.json"

    session = _reset_profiling_output_path
    session.debug.profiling_output_path = str(bad)
    session.profiling.start()
    session.profiling.stop()

    with pytest.raises(ProfilingError) as excinfo:
        session.profiling.wait_for_trace()
    # The message includes the resolved path so users can diagnose without
    # rerunning the workload.
    assert str(bad) in str(excinfo.value)


@_skip_without_recording
def test_wait_for_trace_succeeds_on_happy_path(
    tmp_path: Path, _reset_profiling_output_path: InferenceSession
) -> None:
    # A writable path + a clean stop() must leave wait_for_trace() exception-
    # free and the JSON must be parseable.  This is the inverse of the
    # unwritable-path test above; together they pin the contract on both
    # sides.
    target = tmp_path / "happy-trace.json"

    _capture_to(_reset_profiling_output_path, str(target))  # must not raise

    assert target.exists()
    json.loads(target.read_text())
