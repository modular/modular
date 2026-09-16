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
"""Tests the ModuleV3 production validator."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator

import numpy as np
import pytest
from max.driver import CPU, Accelerator, accelerator_count
from max.dtype import DType
from max.experimental import _validation_hooks as hooks
from max.experimental import functional as F
from max.experimental import realization_context
from max.experimental.compilation import CompiledCallable, compile
from max.experimental.sharding import NoReshard, mode
from max.experimental.sharding.mode import current_solver
from max.experimental.tensor import Tensor
from max.experimental.validation import ProductionValidator
from max.graph import DeviceRef, TensorType
from max.pipelines.modeling.production_validation import prod_validator

_F32 = DType.float32


def _spec(*shape: int) -> TensorType:
    return TensorType(_F32, list(shape), device=DeviceRef.CPU())


def _ones(*shape: int) -> Tensor:
    return Tensor.ones(list(shape), dtype=_F32, device=CPU())


_VALIDATION_LOGGER = "max.experimental.validation"


class _ValidatorLog:
    """The validator's own warnings, ignoring every other logger's."""

    def __init__(self, caplog: pytest.LogCaptureFixture) -> None:
        self._caplog = caplog

    @property
    def messages(self) -> list[str]:
        return [
            record.getMessage()
            for record in self._caplog.records
            if record.name == _VALIDATION_LOGGER
        ]


@pytest.fixture
def warnings(caplog: pytest.LogCaptureFixture) -> Iterator[_ValidatorLog]:
    """Captures just the validator's warnings."""
    with caplog.at_level(logging.WARNING, logger=_VALIDATION_LOGGER):
        yield _ValidatorLog(caplog)


def _double(x: Tensor) -> Tensor:
    return x * 2


def _negate(x: Tensor) -> Tensor:
    return -x


def test_disabled_validator_installs_no_instrumentation() -> None:
    with ProductionValidator(enabled=False):
        assert not hooks.VALIDATORS.get()
    assert not hooks.VALIDATORS.get()


def test_scope_uninstalls_instrumentation_on_the_way_out() -> None:
    with ProductionValidator():
        assert hooks.VALIDATORS.get()
    assert not hooks.VALIDATORS.get()


def test_a_scope_restores_the_realization_context() -> None:
    original = realization_context._DEFAULT_REALIZATION_CONTEXT

    with ProductionValidator():
        assert realization_context._DEFAULT_REALIZATION_CONTEXT is not original

    assert realization_context._DEFAULT_REALIZATION_CONTEXT is original


def test_prod_validator_binds_to_the_innermost_scope() -> None:
    with ProductionValidator() as validator:
        with prod_validator.allow_eager(reason="checking it binds"):
            assert validator._allow_eager == 1
        assert validator._allow_eager == 0


def test_an_escape_hatch_outside_a_scope_does_nothing(
    warnings: _ValidatorLog,
) -> None:
    """Model code carries hatches whether or not the run is validated."""
    with prod_validator.allow_eager(reason="nothing is validating"):
        x = _ones(2, 2)
        F.add(x, x)
    prod_validator.discard_output(None, reason="also a no-op")

    assert not warnings.messages


def test_eager_work_is_tallied_by_call_site(warnings: _ValidatorLog) -> None:
    with ProductionValidator(label="execution"):
        x = _ones(2, 2)
        for _ in range(3):
            x = F.add(x, x)  # one line, so one site, three times

    assert len(warnings.messages) == 1
    assert "during execution" in warnings.messages[0]
    # Four: the three adds, plus creating the tensor they start from.
    assert "4 eager execution(s)" in warnings.messages[0]
    assert "test_production_validator.py:" in warnings.messages[0]
    assert "x3" in warnings.messages[0]


def test_allow_eager_suppresses_the_eager_finding(
    warnings: _ValidatorLog,
) -> None:
    with ProductionValidator() as validator:
        with validator.allow_eager(reason="input batching, off hot path"):
            x = _ones(2, 2)
            F.add(x, x)

    assert not warnings.messages


def test_a_failed_scope_reports_nothing(warnings: _ValidatorLog) -> None:
    with pytest.raises(RuntimeError, match="boom"):
        with ProductionValidator():
            x = _ones(2, 2)
            F.add(x, x)
            raise RuntimeError("boom")

    assert not warnings.messages


def test_a_graph_break_is_reported_with_its_boundary(
    warnings: _ValidatorLog,
) -> None:
    double = compile(_double)(_spec(2, 2))
    negate = compile(_negate)(_spec(2, 2))

    with ProductionValidator():
        negate(double(_ones(2, 2)))

    assert len(warnings.messages) == 1
    assert "compiled-graph boundary" in warnings.messages[0]
    # `_sanitized_graph_name` strips the leading underscore.
    assert "double -> negate" in warnings.messages[0]


def test_graph_break_hatch_suppresses_the_boundary(
    warnings: _ValidatorLog,
) -> None:
    double = compile(_double)(_spec(2, 2))
    negate = compile(_negate)(_spec(2, 2))

    x = _ones(2, 2)
    with ProductionValidator() as validator:
        with validator.graph_break(reason="two-model ensemble"):
            out = negate(double(x))
        validator.discard_output(out, reason="the test never reads it")

    assert not warnings.messages


def test_an_output_nothing_reads_is_reported(warnings: _ValidatorLog) -> None:
    double = compile(_double)(_spec(2, 2))

    with ProductionValidator():
        double(_ones(2, 2))

    assert len(warnings.messages) == 1
    assert "never read" in warnings.messages[0]


def test_an_output_a_later_op_reads_is_not_reported(
    warnings: _ValidatorLog,
) -> None:
    double = compile(_double)(_spec(2, 2))

    x, y = _ones(2, 2), _ones(2, 2)
    with ProductionValidator() as validator:
        with validator.allow_eager(reason="consumption, not eagerness"):
            F.add(double(x), y)

    assert not warnings.messages


def test_discard_output_excuses_a_named_output(warnings: _ValidatorLog) -> None:
    double = compile(_double)(_spec(2, 2))

    x = _ones(2, 2)
    with ProductionValidator() as validator:
        out = double(x)
        validator.discard_output(out, reason="a debug tap")

    assert not warnings.messages


def test_an_output_read_through_dlpack_is_not_reported_unused(
    warnings: _ValidatorLog,
) -> None:
    """``np.from_dlpack`` is how the diffusion path reads a decoded image."""
    double = compile(_double)(_spec(2, 2))

    x = _ones(2, 2)
    with ProductionValidator():
        np.from_dlpack(double(x))

    assert not warnings.messages


def test_tensor_creation_counts_as_eager_work(
    warnings: _ValidatorLog,
) -> None:
    """Allocating per request is a graph compile and launch like any other."""
    with ProductionValidator():
        _ones(4, 4)
        _ones(8, 8)

    assert len(warnings.messages) == 1
    assert "2 eager execution(s)" in warnings.messages[0]


def test_a_batched_region_counts_once(warnings: _ValidatorLog) -> None:
    """``@F.functional`` fuses its body into one compile and one launch."""

    @F.functional
    def a_block(t: Tensor) -> Tensor:
        return F.relu(F.mul(F.add(t, t), t))

    with ProductionValidator() as validator:
        t = _ones(2, 2)
        with validator.allow_eager(reason="not what this checks"):
            pass
        a_block(t)

    assert len(warnings.messages) == 1
    # Three ops, but one realization: the region is what costs a launch.
    assert "2 eager execution(s)" in warnings.messages[0]


def test_a_scope_leaves_the_sharding_solver_alone() -> None:
    """Entering a scope must not change what a distributed model compiles to."""
    outside = current_solver()
    with ProductionValidator():
        assert type(current_solver()) is type(outside)
    with mode(NoReshard()):
        with ProductionValidator():
            assert isinstance(current_solver(), NoReshard)


def test_a_host_side_item_is_not_a_transfer() -> None:
    with ProductionValidator():
        assert Tensor.ones([1], dtype=_F32, device=CPU()).item() == 1.0


@pytest.mark.skipif(not accelerator_count(), reason="needs an accelerator")
def test_a_device_to_host_transfer_is_reported(
    warnings: _ValidatorLog,
) -> None:
    with ProductionValidator():
        Tensor.ones([1], dtype=_F32, device=Accelerator()).item()

    assert len(warnings.messages) == 1
    assert "device-to-host transfer(s)" in warnings.messages[0]
    assert "Tensor.item()" in warnings.messages[0]


@pytest.mark.skipif(not accelerator_count(), reason="needs an accelerator")
def test_allow_device_transfer_permits_the_copy() -> None:
    with ProductionValidator() as validator:
        with validator.allow_device_transfer(reason="one bulk read"):
            value = Tensor.ones([1], dtype=_F32, device=Accelerator()).item()
    assert value == 1.0


def test_a_reentered_scope_reports_once(
    warnings: _ValidatorLog,
) -> None:
    validator = ProductionValidator()

    x = _ones(2, 2)
    with validator:
        with validator:
            F.add(x, x)
        assert hooks.VALIDATORS.get()
        assert not warnings.messages

    assert len(warnings.messages) == 1
    # Not x2: re-entering must not install the instrumentation twice.
    assert "1 eager execution(s)" in warnings.messages[0]


def test_enabled_is_fixed_at_construction() -> None:
    """A scope cannot be turned off from inside itself and left installed."""
    validator = ProductionValidator()
    with pytest.raises(AttributeError):
        validator.enabled = False  # type: ignore[misc]


def _compile_scale(scale: float) -> CompiledCallable[..., Tensor]:
    """Compiles a graph named ``forward``, whatever the scale."""

    def forward(x: Tensor) -> Tensor:
        return x * scale

    return compile(forward)(_spec(2, 2))


def test_graphs_sharing_a_name_are_told_apart(
    warnings: _ValidatorLog,
) -> None:
    """``Model.name`` is the traced callable's ``__name__``, not an identity."""
    doubled, tripled = _compile_scale(7.0), _compile_scale(11.0)

    x = _ones(2, 2)
    with ProductionValidator() as validator:
        out = tripled(doubled(x))
        validator.discard_output(out, reason="the boundary is what this checks")

    assert len(warnings.messages) == 1
    assert "forward#1 -> forward#2" in warnings.messages[0]


def test_eager_work_inside_a_running_loop_is_tallied(
    warnings: _ValidatorLog,
) -> None:
    """Realization hops to a worker thread, which starts with no context."""

    async def work() -> None:
        x = _ones(2, 2)
        F.add(x, x)

    with ProductionValidator():
        asyncio.run(work())

    assert len(warnings.messages) == 1
    assert "2 eager execution(s)" in warnings.messages[0]
