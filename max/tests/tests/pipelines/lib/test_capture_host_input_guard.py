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
"""Tests the captured-host-input guard.

The guard catches a host-resident graph input whose value changed between
capture and replay, which means any allocation size or launch extent an op
derived from it on the host is frozen at the captured value, because replay
re-launches recorded nodes and runs no host code.

These cover the mode parsing and the comparison that produces the diagnostic.
The call site in ``ServeGraphCaptureRunner.replay`` is exercised in
``max/tests/integration/pipelines/test_device_graph_capture.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import Buffer
from max.pipelines.lib.graph_capture import (
    _HOST_INPUT_GUARD_ENV,
    _host_input_diagnostic,
    _resolve_host_input_guard_mode,
)


def _int32(value: int) -> Buffer:
    """A one-element host int32 buffer, the shape of ``batch_context_length``."""
    return Buffer.from_numpy(np.array([value], dtype=np.int32))


def test_unchanged_host_input_is_silent() -> None:
    """A legitimate replay -- the value is the one captured -- reports nothing.

    This is the half that keeps the guard usable: an input read only through a
    device pointer is refreshed correctly and must not be flagged just for
    being present.
    """
    assert (
        _host_input_diagnostic(
            "batch_context_length", 3, _int32(128), _int32(128)
        )
        is None
    )


def test_changed_host_input_is_reported_with_its_name() -> None:
    """The defect case: the diagnostic names the input and both values."""
    diagnostic = _host_input_diagnostic(
        "batch_context_length", 3, _int32(128), _int32(4096)
    )
    assert diagnostic is not None
    assert "batch_context_length" in diagnostic
    assert "index 3" in diagnostic
    assert "128" in diagnostic
    assert "4096" in diagnostic


def test_changed_dtype_or_shape_is_reported() -> None:
    """A changed dtype or shape is a mismatch, not a comparison error."""
    wider = Buffer.from_numpy(np.array([128], dtype=np.int64))
    assert _host_input_diagnostic("scalar", 0, _int32(128), wider) is not None

    longer = Buffer.from_numpy(np.array([128, 128], dtype=np.int32))
    assert _host_input_diagnostic("scalar", 0, _int32(128), longer) is not None


def test_multi_element_host_input_compares_elementwise() -> None:
    """Equality is over the whole buffer, not just its first element."""
    captured = Buffer.from_numpy(np.array([1, 2, 3], dtype=np.int32))
    same = Buffer.from_numpy(np.array([1, 2, 3], dtype=np.int32))
    differs_in_tail = Buffer.from_numpy(np.array([1, 2, 4], dtype=np.int32))

    assert _host_input_diagnostic("offsets", 0, captured, same) is None
    assert _host_input_diagnostic("offsets", 0, captured, differs_in_tail)


def test_guard_is_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset means off: replay pays nothing for a guard nobody asked for."""
    monkeypatch.delenv(_HOST_INPUT_GUARD_ENV, raising=False)
    assert _resolve_host_input_guard_mode() is None


@pytest.mark.parametrize("value", ["", "0", "off", "false", "no", "OFF"])
def test_recognized_off_values(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(_HOST_INPUT_GUARD_ENV, value)
    assert _resolve_host_input_guard_mode() is None


@pytest.mark.parametrize("value", ["report", "abort", "  ABORT  "])
def test_recognized_modes(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(_HOST_INPUT_GUARD_ENV, value)
    assert _resolve_host_input_guard_mode() == value.strip().lower()


def test_unrecognized_mode_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A typo disables a guard silently unless it is rejected loudly."""
    monkeypatch.setenv(_HOST_INPUT_GUARD_ENV, "warn")
    with pytest.raises(ValueError, match="Invalid"):
        _resolve_host_input_guard_mode()


def test_diagnostic_states_its_own_limits() -> None:
    """The docstring must not let a reader take silence for proof.

    A guard described as stronger than it is becomes a reason not to look,
    which is how this defect class stayed hidden. Asserting on the docstring is
    unusual, but the honesty of that text is the feature.
    """
    doc = _host_input_diagnostic.__doc__ or ""
    assert "candidate" in doc
    assert "silence is not proof" in doc
