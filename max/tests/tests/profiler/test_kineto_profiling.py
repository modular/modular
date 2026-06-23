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

"""Smoke tests for the ``session.profiling`` control surface (MXTOOLS-190).

These exercise the public surface that is already final: the disabled
default, start/stop transitions, and the state vocabulary. HTA-import and
Dynolog handshake coverage land once libkineto is Bazelified (MXTOOLS-190).
"""

from collections.abc import Iterator

import pytest
from max._core.profiler import kineto_disable, kineto_is_enabled
from max.driver import CPU
from max.engine import InferenceSession


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


def _new_session() -> InferenceSession:
    return InferenceSession(devices=[CPU()])


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
