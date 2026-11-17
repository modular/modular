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
"""Tests for ModuleV3 MoE layers and the ``max.nn.kernels`` ops they use.

Covers MoE, TensorParallelMoE and ExpertParallelMoE, and the sharding rules
of the MoE kernels in ``max.nn.kernels`` (``moe_finalize``,
``router_gate_mixed_gemv``, ``moe_sigmoid_gemv_router``) that those layers
and their routers run on.
"""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest
from max import functional as F
from max.driver import Device
from max.dtype import DType
from max.graph import Graph
from max.nn.common_layers.mesh_axis import TP
from max.nn.common_layers.moe import (
    MoE,
    TensorParallelMoE,
)
from max.nn.kernels import (
    moe_finalize,
    moe_sigmoid_gemv_router,
    router_gate_mixed_gemv,
)
from max.tensor import Tensor, default_device
from max.tensor.sharding import (
    DeviceMapping,
    DeviceMesh,
    Partial,
    Placement,
    Replicated,
    Sharded,
    ShardingError,
)

# Small dimensions for fast graph-trace tests.
_HIDDEN_DIM = 256
_NUM_EXPERTS = 8
_NUM_EXPERTS_PER_TOKEN = 2
_MOE_DIM = 128
_SEQ_LEN = 4
# The fused router's shape check takes a multiple of the warp width (32 on
# NVIDIA, 64 on AMD) as the expert count, so 64 passes it on any runner.
_ROUTER_EXPERTS = 64


def _make_fake_gpu(id: int = 0) -> MagicMock:
    label = "gpu"
    fake = MagicMock(spec=Device)
    fake.id = id
    fake.label = label
    fake.__eq__ = MagicMock(  # type: ignore[method-assign]
        side_effect=lambda other: (
            getattr(other, "id", None) == id
            and getattr(other, "label", None) == label
        )
    )
    fake.__hash__ = MagicMock(return_value=hash((id, label)))  # type: ignore[method-assign]
    return fake


@pytest.fixture(autouse=True)
def mock_accelerator() -> Iterator[MagicMock]:
    with patch("max.graph.type.Accelerator") as mock:
        mock.side_effect = _make_fake_gpu
        yield mock


def test_layer(mock_accelerator: MagicMock) -> None:
    """Traces a single-device MoE in a lazy context and verifies output shape and device."""
    devices = [mock_accelerator()]

    with F.lazy():
        layer = MoE(
            hidden_dim=_HIDDEN_DIM,
            num_experts=_NUM_EXPERTS,
            num_experts_per_token=_NUM_EXPERTS_PER_TOKEN,
            moe_dim=_MOE_DIM,
        ).to(devices[0])

        gate_up_proj = layer.gate_up_proj
        assert len(gate_up_proj) == 1
        assert list(gate_up_proj[0].shape) == [
            _NUM_EXPERTS,
            2 * _MOE_DIM,
            _HIDDEN_DIM,
        ]
        assert gate_up_proj[0].device == devices[0]

        down_proj = layer.down_proj
        assert len(down_proj) == 1
        assert list(down_proj[0].shape) == [_NUM_EXPERTS, _HIDDEN_DIM, _MOE_DIM]
        assert down_proj[0].device == devices[0]

        x = Tensor.zeros([_SEQ_LEN, _HIDDEN_DIM], device=devices[0])
        out = layer(x)

    assert list(out.shape) == [_SEQ_LEN, _HIDDEN_DIM]
    assert out.device == devices[0]


def test_tensor_parallel_layer(mock_accelerator: MagicMock) -> None:
    """Traces a TensorParallelMoE in a lazy context and verifies output mapping."""
    with F.lazy():
        devices = [mock_accelerator(0), mock_accelerator(1)]
        num_devices = len(devices)
        mesh = DeviceMesh(tuple(devices), (num_devices,), (TP,))
        replicated_mapping = DeviceMapping(mesh, (Replicated(),))

        with default_device(mesh):
            layer = TensorParallelMoE(
                hidden_dim=_HIDDEN_DIM,
                num_experts=_NUM_EXPERTS,
                num_experts_per_token=_NUM_EXPERTS_PER_TOKEN,
                moe_dim=_MOE_DIM,
            )

        # Each device holds a slice of every expert: 2*moe_dim is split along
        # the output axis by num_devices. The weights are a per-device bundle
        # of single-device tensors, one per mesh device.
        gate_up_proj = layer.gate_up_proj
        assert len(gate_up_proj) == num_devices
        for i, shard in enumerate(gate_up_proj):
            assert list(shard.shape) == [
                _NUM_EXPERTS,
                2 * _MOE_DIM // num_devices,
                _HIDDEN_DIM,
            ]
            assert shard.device == devices[i]

        # down_proj weight is [hidden_dim, moe_dim]; shard_and_stack splits the
        # last axis (moe_dim, the contraction dim) by num_devices.
        down_proj = layer.down_proj
        assert len(down_proj) == num_devices
        for i, shard in enumerate(down_proj):
            assert list(shard.shape) == [
                _NUM_EXPERTS,
                _HIDDEN_DIM,
                _MOE_DIM // num_devices,
            ]
            assert shard.device == devices[i]

        x = Tensor.zeros([_SEQ_LEN, _HIDDEN_DIM], device=replicated_mapping)
        out = layer(x)

    assert list(out.shape) == [_SEQ_LEN, _HIDDEN_DIM]
    assert out.mapping.mesh == mesh
    # Output is a partial sum that must be all-reduced across TP ranks.
    assert out.mapping.placements == (Partial(),)


@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("placement", [Replicated(), Sharded(1), Partial()])
def test_moe_finalize_keeps_down_placement(
    mock_accelerator: MagicMock, placement: Placement, bounded: bool
) -> None:
    """The top-k combine is linear in the expert outputs, so it keeps their
    hidden-axis sharding and passes a partial sum through. A replicated
    ``valid_rows`` bound selects ``moe.finalize.bounded`` and keeps the same
    placement rule."""
    with F.lazy():
        devices = [mock_accelerator(0), mock_accelerator(1)]
        mesh = DeviceMesh(tuple(devices), (len(devices),), (TP,))
        replicated = DeviceMapping(mesh, (Replicated(),))
        rows = _SEQ_LEN * _NUM_EXPERTS_PER_TOKEN

        down = Tensor.zeros([rows, _HIDDEN_DIM], device=replicated)
        down = down.rebind_mapping(DeviceMapping(mesh, (placement,)))
        restore = Tensor.zeros([rows], dtype=DType.uint32, device=replicated)
        weights = Tensor.zeros(
            [_SEQ_LEN, _NUM_EXPERTS_PER_TOKEN], device=replicated
        )
        valid_rows = (
            Tensor.zeros([1], dtype=DType.uint32, device=replicated)
            if bounded
            else None
        )
        out = moe_finalize(down, restore, weights, DType.bfloat16, valid_rows)
        ir = str(Graph.current._mlir_op)

    assert out.mapping.placements == (placement,)
    assert out.dtype == DType.bfloat16
    assert ir.count('symbol = "moe.finalize.bounded"') == (
        len(devices) if bounded else 0
    )
    assert ir.count('symbol = "moe.finalize"') == (
        0 if bounded else len(devices)
    )


@pytest.mark.parametrize("placement", [Replicated(), Sharded(0)])
def test_router_gate_mixed_gemv_takes_the_activation_placement(
    mock_accelerator: MagicMock, placement: Placement
) -> None:
    """The fused gate GEMV reads the activation, so its logits follow it.

    The gate weight is a replicated graph constant, so a row-sharded
    activation keeps its rows: the wrapper stages no gather or reshard of
    the activation ahead of the projection.
    """
    with F.lazy():
        devices = [mock_accelerator(0), mock_accelerator(1)]
        mesh = DeviceMesh(tuple(devices), (len(devices),), (TP,))
        replicated = DeviceMapping(mesh, (Replicated(),))

        hidden = Tensor.zeros(
            [_SEQ_LEN, _HIDDEN_DIM], dtype=DType.bfloat16, device=replicated
        )
        hidden = hidden.rebind_mapping(DeviceMapping(mesh, (placement,)))
        weight = Tensor.zeros(
            [_NUM_EXPERTS, _HIDDEN_DIM], dtype=DType.float32, device=replicated
        )
        logits = router_gate_mixed_gemv(hidden, weight)

    assert logits.mapping.mesh == mesh
    assert logits.mapping.placements == (placement,)
    assert logits.dtype == DType.float32
    assert logits.shape.rank == 2
    assert logits.shape[1] == _NUM_EXPERTS
    # The rows follow the activation, not just the placement tag.
    assert logits.shape[0] == hidden.shape[0]


def test_router_gate_mixed_gemv_rejects_a_sharded_gate_weight(
    mock_accelerator: MagicMock,
) -> None:
    """A gate weight split by expert is the one placement the rule refuses.

    Each device would hold its own experts' logits, which is not the
    activation's placement and not a partial sum either, so no row keeps the
    weight in place: satisfying one gathers it, and the default policy
    refuses that automatic move instead of labelling the result wrongly.
    """
    with F.lazy():
        devices = [mock_accelerator(0), mock_accelerator(1)]
        mesh = DeviceMesh(tuple(devices), (len(devices),), (TP,))
        replicated = DeviceMapping(mesh, (Replicated(),))

        hidden = Tensor.zeros(
            [_SEQ_LEN, _HIDDEN_DIM], dtype=DType.bfloat16, device=replicated
        )
        weight = Tensor.zeros(
            [_NUM_EXPERTS, _HIDDEN_DIM],
            dtype=DType.float32,
            device=DeviceMapping(mesh, (Sharded(0),)),
        )
        with pytest.raises(
            ShardingError,
            match=(
                "needs a resharding; this region does not reshard automatically"
            ),
        ):
            router_gate_mixed_gemv(hidden, weight)


def _route_on_two_devices(
    mock_accelerator: MagicMock,
    *,
    placement: Placement,
    weight_placement: Placement,
    scores_dtype: DType,
) -> tuple[DeviceMesh, Tensor, tuple[Tensor, Tensor]]:
    """Stages the fused router over two devices on a bf16 gate."""
    devices = [mock_accelerator(0), mock_accelerator(1)]
    mesh = DeviceMesh(tuple(devices), (len(devices),), (TP,))
    replicated = DeviceMapping(mesh, (Replicated(),))
    hidden = Tensor.zeros(
        [_SEQ_LEN, _HIDDEN_DIM], dtype=DType.bfloat16, device=replicated
    )
    hidden = hidden.rebind_mapping(DeviceMapping(mesh, (placement,)))
    weight = Tensor.zeros(
        [_ROUTER_EXPERTS, _HIDDEN_DIM],
        dtype=DType.bfloat16,
        device=DeviceMapping(mesh, (weight_placement,)),
    )
    bias = Tensor.zeros(
        [_ROUTER_EXPERTS], dtype=DType.float32, device=replicated
    )
    # The op refuses an Apple GPU, and it asks the host, not the mock devices.
    with patch("max.nn._graph_ops._is_apple_gpu", return_value=False):
        routing = moe_sigmoid_gemv_router(
            hidden,
            weight,
            bias,
            _NUM_EXPERTS_PER_TOKEN,
            norm_weights=True,
            routed_scaling_factor=1.0,
            scores_dtype=scores_dtype,
        )
    return mesh, hidden, routing


@pytest.mark.parametrize("placement", [Replicated(), Sharded(0)])
def test_moe_sigmoid_gemv_router_takes_the_activation_placement(
    mock_accelerator: MagicMock, placement: Placement
) -> None:
    """The fused router routes each row it reads, so its outputs follow the
    activation.

    The gate weight and the correction bias are replicated, so a row-sharded
    activation keeps its rows: the wrapper stages no gather or reshard of the
    activation ahead of the router.
    """
    with F.lazy():
        mesh, hidden, routing = _route_on_two_devices(
            mock_accelerator,
            placement=placement,
            weight_placement=Replicated(),
            scores_dtype=DType.float32,
        )

    expert_ids, expert_weights = routing
    for out, dtype in (
        (expert_ids, DType.int32),
        (expert_weights, DType.float32),
    ):
        assert out.mapping.mesh == mesh
        assert out.mapping.placements == (placement,)
        assert out.dtype == dtype
        assert out.shape.rank == 2
        assert out.shape[0] == hidden.shape[0]
        assert out.shape[1] == _NUM_EXPERTS_PER_TOKEN


def test_moe_sigmoid_gemv_router_rejects_a_sharded_gate_weight(
    mock_accelerator: MagicMock,
) -> None:
    """A gate weight split by expert is a placement the rule refuses.

    Each device would hold the scores of its own experts only, which the
    top-k cannot rank. No row keeps the weight in place, so satisfying one
    gathers it, and the default policy refuses that automatic move.
    """
    with (
        F.lazy(),
        pytest.raises(
            ShardingError,
            match=(
                "needs a resharding; this region does not reshard automatically"
            ),
        ),
    ):
        _route_on_two_devices(
            mock_accelerator,
            placement=Replicated(),
            weight_placement=Sharded(0),
            scores_dtype=DType.float32,
        )


def test_moe_sigmoid_gemv_router_rejects_float16_scores_for_a_bf16_gate(
    mock_accelerator: MagicMock,
) -> None:
    """The scores take the gate weight's dtype or float32, nothing else."""
    with F.lazy(), pytest.raises(ValueError, match="scores_dtype"):
        _route_on_two_devices(
            mock_accelerator,
            placement=Replicated(),
            weight_placement=Replicated(),
            scores_dtype=DType.float16,
        )
