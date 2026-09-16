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
"""Tests the ModuleV3 eager usage validator."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator

import numpy as np
import pytest
from max import _validation_hooks as hooks
from max.driver import CPU, Accelerator, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.experimental import functional as F
from max.experimental import realization_context
from max.experimental.compilation import CompiledCallable, compile
from max.experimental.sharding import NoReshard, mode
from max.experimental.sharding.mode import current_solver
from max.experimental.tensor import Tensor
from max.experimental.validation import EagerUsageValidator
from max.graph import DeviceRef, Graph, TensorType
from max.pipelines.modeling.eager_validation import eager_validator

_F32 = DType.float32


def _spec(*shape: int) -> TensorType:
    return TensorType(_F32, list(shape), device=DeviceRef.CPU())


def _ones(*shape: int) -> Tensor:
    return Tensor.ones(list(shape), dtype=_F32, device=CPU())


_VALIDATION_LOGGER = "max.pipelines"


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
    with EagerUsageValidator(enabled=False):
        assert not hooks.VALIDATORS.get()
    assert not hooks.VALIDATORS.get()


def test_scope_uninstalls_instrumentation_on_the_way_out() -> None:
    with EagerUsageValidator():
        assert hooks.VALIDATORS.get()
    assert not hooks.VALIDATORS.get()


def test_a_scope_restores_the_realization_context() -> None:
    original = realization_context._DEFAULT_REALIZATION_CONTEXT

    with EagerUsageValidator():
        assert realization_context._DEFAULT_REALIZATION_CONTEXT is not original

    assert realization_context._DEFAULT_REALIZATION_CONTEXT is original


def test_eager_validator_binds_to_the_innermost_scope() -> None:
    with EagerUsageValidator() as validator:
        with eager_validator.allow_eager(reason="checking it binds"):
            assert validator._allow_eager == 1
        assert validator._allow_eager == 0


def test_an_escape_hatch_outside_a_scope_does_nothing(
    warnings: _ValidatorLog,
) -> None:
    """Model code carries hatches whether or not the run is validated."""
    with eager_validator.allow_eager(reason="nothing is validating"):
        x = _ones(2, 2)
        F.add(x, x)
    eager_validator.discard_output(None, reason="also a no-op")

    assert not warnings.messages


def test_eager_work_is_tallied_by_call_site(warnings: _ValidatorLog) -> None:
    with EagerUsageValidator(label="execution"):
        x = _ones(2, 2)
        for _ in range(3):
            x = F.add(x, x)  # one line, so one site, three times

    assert len(warnings.messages) == 1
    assert "during execution" in warnings.messages[0]
    # Four: the three adds, plus creating the tensor they start from.
    assert "4 eager execution(s)" in warnings.messages[0]
    assert "test_eager_usage_validator.py:" in warnings.messages[0]
    assert "x3" in warnings.messages[0]


def test_allow_eager_suppresses_the_eager_finding(
    warnings: _ValidatorLog,
) -> None:
    with EagerUsageValidator() as validator:
        with validator.allow_eager(reason="input batching, off hot path"):
            x = _ones(2, 2)
            F.add(x, x)

    assert not warnings.messages


def test_a_failed_scope_reports_nothing(warnings: _ValidatorLog) -> None:
    with pytest.raises(RuntimeError, match="boom"):
        with EagerUsageValidator():
            x = _ones(2, 2)
            F.add(x, x)
            raise RuntimeError("boom")

    assert not warnings.messages


def test_a_graph_break_is_reported_with_its_boundary(
    warnings: _ValidatorLog,
) -> None:
    double = compile(_double)(_spec(2, 2))
    negate = compile(_negate)(_spec(2, 2))

    with EagerUsageValidator():
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
    with EagerUsageValidator() as validator:
        with validator.graph_break(reason="two-model ensemble"):
            out = negate(double(x))
        validator.discard_output(out, reason="the test never reads it")

    assert not warnings.messages


def test_an_output_nothing_reads_is_reported(warnings: _ValidatorLog) -> None:
    double = compile(_double)(_spec(2, 2))

    x = _ones(2, 2)
    with EagerUsageValidator():
        double(x)

    assert len(warnings.messages) == 1
    assert "never read" in warnings.messages[0]
    # The reader needs the call that produced it, not a frame inside the
    # validator's own plumbing.
    assert "test_eager_usage_validator.py" in warnings.messages[0]


def test_an_output_a_later_op_reads_is_not_reported(
    warnings: _ValidatorLog,
) -> None:
    double = compile(_double)(_spec(2, 2))

    x, y = _ones(2, 2), _ones(2, 2)
    with EagerUsageValidator() as validator:
        with validator.allow_eager(reason="consumption, not eagerness"):
            F.add(double(x), y)

    assert not warnings.messages


def test_an_output_a_graph_api_model_reads_is_not_reported(
    warnings: _ValidatorLog,
) -> None:
    """The pipelines' token sampler is a graph-API model reading ModuleV3 logits.

    Nothing reports that read but the engine's own execute path, so without it
    every ModuleV3 text pipeline looked like it produced logits nothing wanted.
    """
    double = compile(_double)(_spec(2, 2))
    with Graph("sampler", input_types=[_spec(2, 2)]) as graph:
        graph.output(-graph.inputs[0].tensor)
    sampler = InferenceSession().load(graph)

    x = _ones(2, 2)
    with EagerUsageValidator():
        sampler(double(x).driver_tensor)

    assert not warnings.messages


def test_a_hatch_on_the_first_graph_suppresses_the_boundary(
    warnings: _ValidatorLog,
) -> None:
    """Declaring the graph that ends a chain is enough to excuse the crossing.

    The boundary is reported at the call that *enters* the second graph, so a
    hatch around the first looks like it should miss it. It does not: leaving
    the block starts a fresh chain. This is the shape a conditional
    sub-model wants, as in the gemma3 vision tower, where the second call is
    unconditional and should not be wrapped.
    """
    double = compile(_double)(_spec(2, 2))
    negate = compile(_negate)(_spec(2, 2))

    x = _ones(2, 2)
    with EagerUsageValidator() as validator:
        with validator.graph_break(reason="the tower is its own graph"):
            mid = double(x)
        out = negate(mid)
        validator.discard_output(out, reason="the test never reads it")

    assert not warnings.messages


def test_discard_output_excuses_a_named_output(warnings: _ValidatorLog) -> None:
    double = compile(_double)(_spec(2, 2))

    x = _ones(2, 2)
    with EagerUsageValidator() as validator:
        out = double(x)
        validator.discard_output(out, reason="a debug tap")

    assert not warnings.messages


def test_an_output_read_through_dlpack_is_not_reported_unused(
    warnings: _ValidatorLog,
) -> None:
    """``np.from_dlpack`` is how the diffusion path reads a decoded image."""
    double = compile(_double)(_spec(2, 2))

    x = _ones(2, 2)
    with EagerUsageValidator():
        np.from_dlpack(double(x))

    assert not warnings.messages


def test_tensor_creation_counts_as_eager_work(
    warnings: _ValidatorLog,
) -> None:
    """Allocating per request is a graph compile and launch like any other."""
    with EagerUsageValidator():
        _ones(4, 4)
        _ones(8, 8)

    assert len(warnings.messages) == 1
    assert "2 eager execution(s)" in warnings.messages[0]


def test_a_batched_region_counts_once(warnings: _ValidatorLog) -> None:
    """``@F.functional`` fuses its body into one compile and one launch."""

    @F.functional
    def a_block(t: Tensor) -> Tensor:
        return F.relu(F.mul(F.add(t, t), t))

    with EagerUsageValidator() as validator:
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
    with EagerUsageValidator():
        assert type(current_solver()) is type(outside)
    with mode(NoReshard()):
        with EagerUsageValidator():
            assert isinstance(current_solver(), NoReshard)


def test_a_host_side_item_is_not_a_transfer() -> None:
    with EagerUsageValidator():
        assert Tensor.ones([1], dtype=_F32, device=CPU()).item() == 1.0


@pytest.mark.skipif(not accelerator_count(), reason="needs an accelerator")
def test_a_device_to_host_transfer_is_reported(
    warnings: _ValidatorLog,
) -> None:
    with EagerUsageValidator():
        Tensor.ones([1], dtype=_F32, device=Accelerator()).item()

    assert len(warnings.messages) == 1
    assert "device-to-host transfer(s)" in warnings.messages[0]
    assert "Tensor.item()" in warnings.messages[0]


@pytest.mark.skipif(not accelerator_count(), reason="needs an accelerator")
def test_allow_device_transfer_permits_the_copy() -> None:
    with EagerUsageValidator() as validator:
        with validator.allow_device_transfer(reason="one bulk read"):
            value = Tensor.ones([1], dtype=_F32, device=Accelerator()).item()
    assert value == 1.0


def test_a_reentered_scope_reports_once(
    warnings: _ValidatorLog,
) -> None:
    validator = EagerUsageValidator()

    x = _ones(2, 2)
    with validator:
        with validator:
            F.add(x, x)
        assert hooks.VALIDATORS.get()
        assert not warnings.messages

    assert len(warnings.messages) == 1
    # Not x2: re-entering must not install the instrumentation twice.
    assert "1 eager execution(s)" in warnings.messages[0]


def _add_at_one_site(x: Tensor) -> Tensor:
    """Adds from a fixed line, so every call reports the same call site."""
    return F.add(x, x)


def test_a_site_already_named_is_not_named_again(
    warnings: _ValidatorLog,
) -> None:
    """A pipeline enters one validator per execution; repeats are not news."""
    validator = EagerUsageValidator()
    x = _ones(2, 2)

    with validator:
        _add_at_one_site(x)
    with validator:
        _add_at_one_site(x)

    assert len(warnings.messages) == 1


def test_a_new_site_is_named_however_late_it_turns_up(
    warnings: _ValidatorLog,
) -> None:
    """Silence on the familiar must not cost the report a fresh finding."""
    validator = EagerUsageValidator()
    x = _ones(2, 2)

    with validator:
        _add_at_one_site(x)
    with validator:
        _add_at_one_site(x)
        F.sub(x, x)

    first, second = warnings.messages
    (already_named,) = [
        line for line in first.splitlines() if line.startswith("    ")
    ]
    sites = [line for line in second.splitlines() if line.startswith("    ")]
    assert sites != [already_named]
    assert len(sites) == 1


def test_enabled_is_fixed_at_construction() -> None:
    """A scope cannot be turned off from inside itself and left installed."""
    validator = EagerUsageValidator()
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
    with EagerUsageValidator() as validator:
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

    with EagerUsageValidator():
        asyncio.run(work())

    assert len(warnings.messages) == 1
    assert "2 eager execution(s)" in warnings.messages[0]
