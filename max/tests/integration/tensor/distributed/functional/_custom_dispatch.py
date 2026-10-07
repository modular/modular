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
"""Shared test logic for custom op dispatch.

Makes a custom kernel distribution-aware: the rule picks the placements,
``transfer_to`` moves the inputs there, and ``call_on_mesh`` runs the
kernel on each device.

DO NOT run this file directly — it contains base classes that are
subclassed by test_custom_dispatch_simulated_cpu.py.

Subclasses must define:
    MESH_2: DeviceMesh  — 2 devices, shape (2,), axis_names=("tp",)
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pytest
from max.experimental import functional as F
from max.experimental import tensor as _tensor_mod
from max.experimental.functional import transfer_to
from max.experimental.sharding import (
    AxisAssignment,
    DeviceMapping,
    DeviceMesh,
    Replicated,
    Sharded,
    ShardingError,
    TensorLayout,
    Unknown,
)
from max.experimental.tensor import Tensor
from max.graph import TensorValue, Value, ops

# ═════════════════════════════════════════════════════════════════════════
#  Shared: placement rule + graph kernel
# ═════════════════════════════════════════════════════════════════════════
# The rule is a pure function on metadata — it could live in
# sharding/rules/ alongside the built-in rules.


def rms_norm_rule(
    x: TensorLayout,
    weight: TensorLayout,
    eps: float = 1e-6,
) -> tuple[
    tuple[DeviceMapping, DeviceMapping, float], tuple[DeviceMapping, ...]
]:
    """RMSNorm reduces over the last dim — cannot be sharded there."""
    placements = x.mapping.placements
    ndim = x.rank
    for p in placements:
        if isinstance(p, Sharded) and p.axis == ndim - 1:
            raise ValueError(
                "rms_norm: cannot shard hidden dim. "
                "Gather first or shard a different axis."
            )
    out_mapping = DeviceMapping(x.mesh, placements)
    return (out_mapping, weight.mapping, eps), (out_mapping,)


def _rms_norm_kernel(
    x: TensorValue,
    weight: TensorValue,
    eps: float = 1e-6,
    weight_offset: float = 0.0,
) -> TensorValue:
    """Single-device rms_norm via the built-in graph op."""
    return ops.rms_norm(x, weight, epsilon=eps, weight_offset=weight_offset)


def rms_norm(
    x: _tensor_mod.Tensor,
    weight: _tensor_mod.Tensor,
    eps: float = 1e-6,
) -> _tensor_mod.Tensor:
    (x_mapping, weight_mapping, eps), (out_mapping,) = rms_norm_rule(
        x.layout, weight.layout, eps
    )
    return F.call_on_mesh(
        _rms_norm_kernel, x_mapping.mesh, out_specs=out_mapping
    )(transfer_to(x, x_mapping), transfer_to(weight, weight_mapping), eps)


# ═════════════════════════════════════════════════════════════════════════
#  Numpy reference
# ═════════════════════════════════════════════════════════════════════════


def _rms_norm_numpy(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    variance = np.mean(x**2, axis=-1, keepdims=True)
    return x / np.sqrt(variance + eps) * w


# ═════════════════════════════════════════════════════════════════════════
#  Test classes
# ═════════════════════════════════════════════════════════════════════════


class _CustomDispatchExplicit:
    """Tests for a custom op that runs through call_on_mesh."""

    MESH_2: ClassVar[DeviceMesh]

    def test_rms_norm_replicated(self) -> None:
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((4, 8)).astype(np.float32)
        w_np = np.ones(8, dtype=np.float32)

        x = transfer_to(
            Tensor(x_np), DeviceMapping(self.MESH_2, (Replicated(),))
        )
        w = transfer_to(
            Tensor(w_np), DeviceMapping(self.MESH_2, (Replicated(),))
        )
        result = rms_norm(x, w, 1e-6)
        assert result.placements == (Replicated(),)
        expected = _rms_norm_numpy(x_np, w_np, 1e-6)
        np.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-4)

    def test_rms_norm_batch_sharded(self) -> None:
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((4, 8)).astype(np.float32)
        w_np = np.ones(8, dtype=np.float32)

        x = transfer_to(Tensor(x_np), DeviceMapping(self.MESH_2, (Sharded(0),)))
        w = transfer_to(
            Tensor(w_np), DeviceMapping(self.MESH_2, (Replicated(),))
        )
        result = rms_norm(x, w, 1e-6)
        assert result.placements == (Sharded(0),)
        expected = _rms_norm_numpy(x_np, w_np, 1e-6)
        np.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-4)


class CustomDispatchTests(_CustomDispatchExplicit):
    """Aggregates all custom dispatch test classes."""

    def test_rebind_mapping_moves_no_data(self) -> None:
        source = transfer_to(
            Tensor(np.ones(4, dtype=np.float32)),
            DeviceMapping(self.MESH_2, (Replicated(),)),
        )
        unknown = source.rebind_mapping(
            DeviceMapping(self.MESH_2, (Unknown(),))
        )
        assert unknown.placements == (Unknown(),)
        assert all(
            left is right
            for left, right in zip(source.buffers, unknown.buffers, strict=True)
        )

    def test_transfer_to_rejects_unknown(self) -> None:
        replicated = DeviceMapping(self.MESH_2, (Replicated(),))
        source = transfer_to(Tensor(np.ones(4, dtype=np.float32)), replicated)
        unknown = DeviceMapping(self.MESH_2, (Unknown(),))
        with pytest.raises(ShardingError, match="rebind_mapping"):
            transfer_to(source, unknown)
        with pytest.raises(ShardingError, match="rebind_mapping"):
            transfer_to(source.rebind_mapping(unknown), replicated)

    def test_rule_less_op_rejects_mixed_meshes(self) -> None:
        other = DeviceMesh(self.MESH_2.devices, (1, 2), ("dp", "tp"))
        x = transfer_to(
            Tensor(np.ones(4, dtype=np.float32)),
            DeviceMapping(self.MESH_2, (Replicated(),)),
        )
        y = transfer_to(
            Tensor(np.ones(4, dtype=np.float32)),
            DeviceMapping(other, (Replicated(), Replicated())),
        )

        def add(a: TensorValue, b: TensorValue) -> TensorValue:
            return a + b

        with pytest.raises(ShardingError, match="meshes of one shape"):
            F.functional(add)(x, y)

    def test_rule_less_op_reads_each_device_rows(self) -> None:
        rows = np.arange(12, dtype=np.float32).reshape(6, 2)
        split = transfer_to(
            Tensor(rows), DeviceMapping(self.MESH_2, (Sharded(0),))
        )

        def first_row(value: TensorValue) -> TensorValue:
            return value[:1]

        result = F.functional(first_row)(split)
        assert result.placements == (Unknown(),)
        result = result.rebind_mapping(split.mapping)
        np.testing.assert_array_equal(result.to_numpy(), rows[[0, 3]])

    def test_op_takes_each_parameter_as_its_hint_says(self) -> None:
        seen: dict[str, type] = {}

        def op(
            x: TensorValue,
            /,
            *rest: TensorValue,
            scale: Tensor,
            **named: Value[Any],
        ) -> TensorValue:
            seen.update(
                x=type(x),
                rest=type(rest[0]),
                scale=type(scale),
                named=type(named["y"]),
            )
            return x + rest[0]

        def rule(
            x: TensorLayout,
            /,
            *rest: TensorLayout,
            scale: TensorLayout,
            **named: TensorLayout,
        ) -> list[AxisAssignment]:
            return [AxisAssignment((Replicated(),) * 4, (Replicated(),))]

        replicated = DeviceMapping(self.MESH_2, (Replicated(),))
        ones = transfer_to(Tensor(np.ones(4, dtype=np.float32)), replicated)
        result = F.functional(op, rule=rule)(ones, ones, scale=ones, y=ones)
        assert seen == {
            "x": TensorValue,
            "rest": TensorValue,
            "scale": Tensor,
            "named": TensorValue,
        }
        assert result.placements == (Replicated(),)
        np.testing.assert_array_equal(result.to_numpy(), np.full(4, 2.0))
