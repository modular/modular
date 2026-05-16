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
"""Tests for MoE, TensorParallelMoE, and ExpertParallelMoE (ModuleV3)."""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest
from max.driver import CPU, Device
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn.common_layers.mesh_axis import TP
from max.experimental.nn.common_layers.moe import (
    ExpertParallelMoE,
    MoE,
    TensorParallelMoE,
)
from max.experimental.sharding import (
    DeviceMesh,
    Partial,
    PlacementMapping,
    Replicated,
)
from max.experimental.tensor import Tensor
from max.graph import BufferValue, TensorValue
from max.nn.comm.ep import EPBatchManager, EPConfig
from max.nn.comm.ep.ep_config import NUM_GROUPS
from max.nn.comm.ep.ep_manager import get_ep_local_sync_counters_size

# Small dimensions for fast graph-trace tests.
_HIDDEN_DIM = 256
_NUM_EXPERTS = 8
_NUM_EXPERTS_PER_TOKEN = 2
_MOE_DIM = 128
_SEQ_LEN = 4


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
        assert list(gate_up_proj.shape) == [
            _NUM_EXPERTS,
            2 * _MOE_DIM,
            _HIDDEN_DIM,
        ]
        assert gate_up_proj.device == devices[0]

        down_proj = layer.down_proj
        assert list(down_proj.shape) == [_NUM_EXPERTS, _HIDDEN_DIM, _MOE_DIM]
        assert down_proj.device == devices[0]

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
        replicated_mapping = PlacementMapping(mesh, (Replicated(),))

        layer = TensorParallelMoE(
            hidden_dim=_HIDDEN_DIM,
            num_experts=_NUM_EXPERTS,
            num_experts_per_token=_NUM_EXPERTS_PER_TOKEN,
            moe_dim=_MOE_DIM,
        ).to(mesh)

        # Each device holds a slice of every expert: 2*moe_dim is split along
        # the output axis by num_devices.
        gate_up_proj = layer.gate_up_proj
        assert list(gate_up_proj.shape) == [
            _NUM_EXPERTS,
            2 * _MOE_DIM // num_devices,
            _HIDDEN_DIM,
        ]
        assert gate_up_proj.mapping.mesh == mesh

        # down_proj weight is [hidden_dim, moe_dim]; shard_and_stack splits the
        # last axis (moe_dim, the contraction dim) by num_devices.
        down_proj = layer.down_proj
        assert list(down_proj.shape) == [
            _NUM_EXPERTS,
            _HIDDEN_DIM,
            _MOE_DIM // num_devices,
        ]
        assert down_proj.mapping.mesh == mesh

        x = Tensor.zeros([_SEQ_LEN, _HIDDEN_DIM], device=replicated_mapping)
        out = layer(x)

    assert list(out.shape) == [_SEQ_LEN, _HIDDEN_DIM]
    assert out.mapping.mesh == mesh
    # Output is a partial sum that must be all-reduced across TP ranks.
    assert out.mapping.to_placements() == (Partial(),)


def _build_ep_batch_manager(
    config: EPConfig, devices: list[Device]
) -> EPBatchManager:
    """Construct an EPBatchManager and attach placeholder buffer values.

    Bypasses :meth:`EPBatchManager.fetch_buffers` so the EP forward path can
    be traced under :func:`F.lazy` without wiring up real graph inputs.
    """
    mgr = EPBatchManager(config)
    n_devices = config.n_gpus_per_node
    n_experts_for_counters = (
        config.n_experts // n_devices
        if config.use_allreduce
        else config.n_experts
    )
    counter_size = get_ep_local_sync_counters_size(n_experts_for_counters)

    # Per-group, per-device atomic counters (BufferValue, int32, on each GPU).
    mgr._atomic_counters = []
    for _ in range(NUM_GROUPS):
        group: list[BufferValue] = []
        for i in range(n_devices):
            buf = Tensor.zeros(
                [counter_size], dtype=DType.int32, device=devices[i]
            )
            group.append(BufferValue(buf))
        mgr._atomic_counters.append(group)

    # Per-group send/recv/recv_count pointer tensors (uint64, on CPU).
    def _make_ptrs() -> list[TensorValue]:
        return [
            TensorValue(
                Tensor.zeros([n_devices], dtype=DType.uint64, device=CPU())
            )
            for _ in range(NUM_GROUPS)
        ]

    mgr._send_buf_ptrs = _make_ptrs()
    mgr._recv_buf_ptrs = _make_ptrs()
    mgr._recv_count_ptrs = _make_ptrs()
    return mgr


def test_expert_parallel_layer(mock_accelerator: MagicMock) -> None:
    """Traces an ExpertParallelMoE in a lazy context and verifies output mapping."""
    with F.lazy():
        devices = [mock_accelerator(0), mock_accelerator(1)]
        num_devices = len(devices)
        num_local_experts = _NUM_EXPERTS // num_devices
        mesh = DeviceMesh(tuple(devices), (num_devices,), (TP,))
        replicated_mapping = PlacementMapping(mesh, (Replicated(),))

        ep_config = EPConfig(
            dispatch_dtype=DType.bfloat16,
            combine_dtype=DType.bfloat16,
            hidden_size=_HIDDEN_DIM,
            top_k=_NUM_EXPERTS_PER_TOKEN,
            n_experts=_NUM_EXPERTS,
            max_tokens_per_rank=_SEQ_LEN,
            n_gpus_per_node=num_devices,
            n_nodes=1,
        )
        ep_batch_manager = _build_ep_batch_manager(ep_config, devices)

        layer = ExpertParallelMoE(
            hidden_dim=_HIDDEN_DIM,
            num_experts=_NUM_EXPERTS,
            num_experts_per_token=_NUM_EXPERTS_PER_TOKEN,
            moe_dim=_MOE_DIM,
            ep_batch_manager=ep_batch_manager,
        ).to(mesh)

        # Each device holds num_local_experts full-size experts.
        gate_up_proj = layer.gate_up_proj
        assert list(gate_up_proj.shape) == [
            num_local_experts,
            2 * _MOE_DIM,
            _HIDDEN_DIM,
        ]
        assert gate_up_proj.mapping.mesh == mesh

        down_proj = layer.down_proj
        assert list(down_proj.shape) == [
            num_local_experts,
            _HIDDEN_DIM,
            _MOE_DIM,
        ]
        assert down_proj.mapping.mesh == mesh

        x = Tensor.zeros([_SEQ_LEN, _HIDDEN_DIM], device=replicated_mapping)
        out = layer(x)

    assert list(out.shape) == [_SEQ_LEN, _HIDDEN_DIM]
    assert out.mapping.mesh == mesh
