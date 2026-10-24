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

"""Lazy-trace tests for TensorParallelMoE and ExpertParallelMoE."""

from __future__ import annotations

import dataclasses
import re
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import pytest
from max.driver import CPU, Device
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn.common_layers.functional_kernels import (
    moe_create_indices,
)
from max.experimental.nn.common_layers.mesh_axis import TP
from max.experimental.nn.common_layers.quant_linear import QuantizedMLP
from max.experimental.nn.common_layers.quant_moe import (
    ExpertParallelMoE,
    TensorParallelMoE,
)
from max.experimental.nn.common_layers.quant_tensor import (
    FP8BlockTensor,
    MXFP8Tensor,
    NVFP4Tensor,
)
from max.experimental.sharding import (
    DeviceMapping,
    DeviceMesh,
    Partial,
    Replicated,
)
from max.experimental.tensor import Tensor, default_device, default_dtype
from max.graph import Graph
from max.nn.comm.ep import EPConfig
from max.nn.comm.ep.ep_config import NUM_GROUPS
from max.nn.comm.ep.ep_manager import (
    EPBatchManager,
    EPCommBuffers,
    get_ep_local_sync_counters_size,
)
from max.nn.quant_config import QuantConfig
from max.pipelines.architectures.deepseekV3_2_modulev3.layers.mlp import (
    DeepseekV3_2MLP,
)
from max.pipelines.architectures.deepseekV3_2_modulev3.layers.moe_gate import (
    DeepseekV3_2TopKRouter,
)
from max.pipelines.architectures.deepseekV3_2_modulev3.layers.transformer_block import (
    _get_mlp,
)
from max.pipelines.architectures.deepseekV3_2_modulev3.model_config import (
    DeepseekV3_2Config,
)
from max.pipelines.architectures.deepseekV3_modulev3.layers.transformer_block import (
    ParallelismMode,
)

_HIDDEN_DIM = 256
_MOE_DIM = 128
_NUM_EXPERTS = 4
_NUM_EXPERTS_PER_TOKEN = 2
_SEQ_LEN = 4

# FP8 block-scaled multi-device dims chosen so the (128, 128) weight-scale grid
# divides evenly across 2 devices on every sharded axis.
_FP8_HIDDEN_DIM = 256
_FP8_MOE_DIM = 512


# --------------------------------------------------------------------------- #
# Distributed dispatch of the MoE index kernel
# --------------------------------------------------------------------------- #


def test_moe_create_indices_forwards_keyword_only_args(
    mock_accelerator: MagicMock,
) -> None:
    """The NVFP4 scale-offset output survives the per-shard dispatch.

    ``needs_scales_offset`` is keyword-only, so the sharded-call path has to
    forward non-tensor keyword arguments to every per-device call; dropping
    them would silently cost the NVFP4 MoE its sixth index output.
    """
    with F.lazy():
        devices = [mock_accelerator(0), mock_accelerator(1)]
        mesh = DeviceMesh(tuple(devices), (len(devices),), (TP,))
        replicated = DeviceMapping(mesh, (Replicated(),))
        topk_ids = Tensor.zeros(
            [_SEQ_LEN * _NUM_EXPERTS_PER_TOKEN],
            dtype=DType.int32,
            device=replicated,
        )

        assert len(moe_create_indices(topk_ids, _NUM_EXPERTS)) == 5

        with_offsets = moe_create_indices(
            topk_ids, _NUM_EXPERTS, needs_scales_offset=True
        )
        assert len(with_offsets) == 6
        assert with_offsets[-1].mapping.mesh == mesh


# --------------------------------------------------------------------------- #
# TensorParallelMoE
# --------------------------------------------------------------------------- #


def test_tensor_parallel_moe_bf16(mock_accelerator: MagicMock) -> None:
    """TP shards every expert's moe_dim; forward returns a Partial sum."""
    with F.lazy():
        devices = [mock_accelerator(0), mock_accelerator(1)]
        num_devices = len(devices)
        mesh = DeviceMesh(tuple(devices), (num_devices,), (TP,))
        replicated = DeviceMapping(mesh, (Replicated(),))

        with default_dtype(DType.bfloat16):
            with default_device(mesh):
                layer = TensorParallelMoE(
                    hidden_dim=_HIDDEN_DIM,
                    num_experts=_NUM_EXPERTS,
                    num_experts_per_token=_NUM_EXPERTS_PER_TOKEN,
                    moe_dim=_MOE_DIM,
                )

            gate_up = layer.gate_up_proj
            assert len(gate_up) == num_devices
            for i, shard in enumerate(gate_up):
                assert isinstance(shard, Tensor)
                assert list(shard.shape) == [
                    _NUM_EXPERTS,
                    2 * _MOE_DIM // num_devices,
                    _HIDDEN_DIM,
                ]
                assert shard.device == devices[i]

            down = layer.down_proj
            assert len(down) == num_devices
            for shard in down:
                assert isinstance(shard, Tensor)
                assert list(shard.shape) == [
                    _NUM_EXPERTS,
                    _HIDDEN_DIM,
                    _MOE_DIM // num_devices,
                ]

            x = Tensor.zeros(
                [_SEQ_LEN, _HIDDEN_DIM],
                dtype=DType.bfloat16,
                device=replicated,
            )
            out = layer(x)

        assert list(out.shape) == [_SEQ_LEN, _HIDDEN_DIM]
        assert out.mapping.mesh == mesh
        # forward returns each device's partial sum; the single all-reduce that
        # resolves it lives in the transformer block after the MoE layer.
        assert out.mapping.placements == (Partial(),)


def test_tensor_parallel_moe_bf16_shared_experts(
    mock_accelerator: MagicMock,
) -> None:
    """Routed + shared experts fold into one Partial sum (single all-reduce)."""
    with F.lazy():
        devices = [mock_accelerator(0), mock_accelerator(1)]
        num_devices = len(devices)
        mesh = DeviceMesh(tuple(devices), (num_devices,), (TP,))
        replicated = DeviceMapping(mesh, (Replicated(),))

        with default_dtype(DType.bfloat16):
            with default_device(mesh):
                layer = TensorParallelMoE(
                    hidden_dim=_HIDDEN_DIM,
                    num_experts=_NUM_EXPERTS,
                    num_experts_per_token=_NUM_EXPERTS_PER_TOKEN,
                    moe_dim=_MOE_DIM,
                    has_shared_experts=True,
                    shared_experts_dim=_MOE_DIM,
                )

            x = Tensor.zeros(
                [_SEQ_LEN, _HIDDEN_DIM],
                dtype=DType.bfloat16,
                device=replicated,
            )
            out = layer(x)

        assert list(out.shape) == [_SEQ_LEN, _HIDDEN_DIM]
        assert out.mapping.mesh == mesh
        # The shared expert is computed inside the per-device local_map and
        # summed with the routed experts, so the layer output is a single
        # Partial sum (one all-reduce resolves both, in the transformer block).
        assert out.mapping.placements == (Partial(),)


def test_tensor_parallel_moe_fp8_weights(
    mock_accelerator: MagicMock, fp8_quant_config: QuantConfig
) -> None:
    """FP8 TP shards both the packed data and the block-scale grid."""
    with F.lazy():
        devices = [mock_accelerator(0), mock_accelerator(1)]
        num_devices = len(devices)
        mesh = DeviceMesh(tuple(devices), (num_devices,), (TP,))
        replicated = DeviceMapping(mesh, (Replicated(),))

        with default_device(mesh):
            layer = TensorParallelMoE(
                hidden_dim=_FP8_HIDDEN_DIM,
                num_experts=_NUM_EXPERTS,
                num_experts_per_token=_NUM_EXPERTS_PER_TOKEN,
                moe_dim=_FP8_MOE_DIM,
                quant_config=fp8_quant_config,
            )

        gate_up = layer.gate_up_proj
        assert isinstance(gate_up, list)
        assert isinstance(gate_up[0], FP8BlockTensor)
        assert list(gate_up[0].weight.shape) == [
            _NUM_EXPERTS,
            2 * _FP8_MOE_DIM // num_devices,
            _FP8_HIDDEN_DIM,
        ]
        # Scale grid is the (128, 128)-block count of the sharded data.
        assert list(gate_up[0].weight_scale_inv.shape) == [
            _NUM_EXPERTS,
            2 * _FP8_MOE_DIM // num_devices // 128,
            _FP8_HIDDEN_DIM // 128,
        ]

        down = layer.down_proj
        assert isinstance(down, list)
        assert isinstance(down[0], FP8BlockTensor)
        assert list(down[0].weight.shape) == [
            _NUM_EXPERTS,
            _FP8_HIDDEN_DIM,
            _FP8_MOE_DIM // num_devices,
        ]
        assert list(down[0].weight_scale_inv.shape) == [
            _NUM_EXPERTS,
            _FP8_HIDDEN_DIM // 128,
            _FP8_MOE_DIM // num_devices // 128,
        ]

        x = Tensor.zeros(
            [_SEQ_LEN, _FP8_HIDDEN_DIM],
            dtype=DType.bfloat16,
            device=replicated,
        )
        out = layer(x)

    assert list(out.shape) == [_SEQ_LEN, _FP8_HIDDEN_DIM]
    assert out.mapping.mesh == mesh
    # The FP8 grouped matmul runs on each device's slice of moe_dim (see
    # quant_moe.TensorParallelMoE.forward), so the layer returns a Partial sum
    # for the caller to all-reduce.
    assert out.mapping.placements == (Partial(),)


# --------------------------------------------------------------------------- #
# ExpertParallelMoE
# --------------------------------------------------------------------------- #


def _build_ep_batch_manager(
    config: EPConfig, devices: list[Device]
) -> tuple[EPBatchManager, EPCommBuffers]:
    """Construct an EPBatchManager and placeholder comm buffers for it.

    Bypasses :meth:`EPBatchManager.comm_buffers` so the EP forward path can be
    traced under :func:`F.lazy` without wiring up real graph inputs. The
    returned buffers are what the caller threads into
    ``ExpertParallelMoE.forward``, which binds them onto the manager.
    """
    n_devices = config.n_gpus_per_node
    n_experts_for_counters = (
        config.n_experts // n_devices
        if config.use_allreduce
        else config.n_experts
    )
    counter_size = get_ep_local_sync_counters_size(n_experts_for_counters)

    atomic_counters = [
        [
            Tensor.zeros([counter_size], dtype=DType.int32, device=devices[i])
            for i in range(n_devices)
        ]
        for _ in range(NUM_GROUPS)
    ]

    def _make_ptrs() -> list[Tensor]:
        return [
            Tensor.zeros([n_devices], dtype=DType.uint64, device=CPU())
            for _ in range(NUM_GROUPS)
        ]

    comm = EPCommBuffers(
        atomic_counters=atomic_counters,
        send_buf_ptrs=_make_ptrs(),
        recv_buf_ptrs=_make_ptrs(),
        recv_count_ptrs=_make_ptrs(),
    )
    return EPBatchManager(config), comm


def _ep_config(dispatch_dtype: DType, num_devices: int, **kwargs) -> EPConfig:
    return EPConfig(
        dispatch_dtype=dispatch_dtype,
        combine_dtype=DType.bfloat16,
        hidden_size=_HIDDEN_DIM,
        top_k=_NUM_EXPERTS_PER_TOKEN,
        n_experts=_NUM_EXPERTS,
        max_tokens_per_rank=_SEQ_LEN,
        n_gpus_per_node=num_devices,
        n_nodes=1,
        **kwargs,
    )


def test_expert_parallel_moe_bf16(mock_accelerator: MagicMock) -> None:
    """EP distributes whole experts; weights keep full moe_dim per device."""
    with F.lazy():
        devices = [mock_accelerator(0), mock_accelerator(1)]
        num_devices = len(devices)
        num_local_experts = _NUM_EXPERTS // num_devices
        mesh = DeviceMesh(tuple(devices), (num_devices,), (TP,))
        replicated = DeviceMapping(mesh, (Replicated(),))

        ep_batch_manager, comm_buffers = _build_ep_batch_manager(
            _ep_config(DType.bfloat16, num_devices), devices
        )

        with default_dtype(DType.bfloat16):
            with default_device(mesh):
                layer = ExpertParallelMoE(
                    hidden_dim=_HIDDEN_DIM,
                    num_experts=_NUM_EXPERTS,
                    num_experts_per_token=_NUM_EXPERTS_PER_TOKEN,
                    moe_dim=_MOE_DIM,
                    ep_batch_manager=ep_batch_manager,
                )

            gate_up = layer.gate_up_proj
            assert len(gate_up) == num_devices
            for i, shard in enumerate(gate_up):
                assert isinstance(shard, Tensor)
                assert list(shard.shape) == [
                    num_local_experts,
                    2 * _MOE_DIM,
                    _HIDDEN_DIM,
                ]
                assert shard.device == devices[i]

            down = layer.down_proj
            assert len(down) == num_devices
            for shard in down:
                assert isinstance(shard, Tensor)
                assert list(shard.shape) == [
                    num_local_experts,
                    _HIDDEN_DIM,
                    _MOE_DIM,
                ]

            x = Tensor.zeros(
                [_SEQ_LEN, _HIDDEN_DIM],
                dtype=DType.bfloat16,
                device=replicated,
            )
            out = layer(x, comm_buffers)

        assert list(out.shape) == [_SEQ_LEN, _HIDDEN_DIM]
        assert out.mapping.mesh == mesh


def test_expert_parallel_moe_allreduce_shared_expert_on_side_stream(
    mock_accelerator: MagicMock,
) -> None:
    """Under allreduce the shared expert runs on rank 0, on a side stream.

    It is the only extra work that rank carries, so on the default stream it
    lands directly on rank 0's critical path and every other rank spins in the
    post-MoE Lamport allreduce waiting for it.
    """
    with F.lazy():
        devices = [mock_accelerator(0), mock_accelerator(1)]
        num_devices = len(devices)
        mesh = DeviceMesh(tuple(devices), (num_devices,), (TP,))
        replicated = DeviceMapping(mesh, (Replicated(),))

        ep_batch_manager, comm_buffers = _build_ep_batch_manager(
            _ep_config(DType.bfloat16, num_devices, use_allreduce=True),
            devices,
        )

        with default_device(mesh), default_dtype(DType.bfloat16):
            layer = ExpertParallelMoE(
                hidden_dim=_HIDDEN_DIM,
                num_experts=_NUM_EXPERTS,
                num_experts_per_token=_NUM_EXPERTS_PER_TOKEN,
                moe_dim=_MOE_DIM,
                has_shared_experts=True,
                shared_experts_dim=_MOE_DIM,
                ep_batch_manager=ep_batch_manager,
            )

            x = Tensor.zeros(
                [_SEQ_LEN, _HIDDEN_DIM],
                dtype=DType.bfloat16,
                device=replicated,
            )
            out = layer(x, comm_buffers)
            ir = str(Graph.current._mlir_op)

    assert list(out.shape) == [_SEQ_LEN, _HIDDEN_DIM]
    # One region, fed by exactly one shard. The MOGG->MGP lowering derives its
    # side-stream views from the region's operands and results, so a single
    # rank-0 operand keeps it to one view instead of one per device.
    assert ir.count('"mo.sequence"') == 1
    assert re.search(r'"mo\.sequence"\(%\w+\) <\{streamId = 1 : ui64\}>', ir)


@pytest.mark.parametrize("overlap_env", ["0", "1"])
def test_expert_parallel_moe_shared_expert_between_dispatch_launch_and_wait(
    mock_accelerator: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    overlap_env: str,
) -> None:
    """Peer-to-peer EP runs the shared expert between dispatch launch and wait.

    This matches the graph-API EP path for any ``MODULAR_OVERLAP_SHARED_EXPERT``
    value: the blocking multi-device dispatch and combine would serialize the
    shared expert behind the EP communication.
    """
    monkeypatch.setenv("MODULAR_OVERLAP_SHARED_EXPERT", overlap_env)
    with F.lazy():
        devices = [mock_accelerator(0), mock_accelerator(1)]
        num_devices = len(devices)
        mesh = DeviceMesh(tuple(devices), (num_devices,), (TP,))
        replicated = DeviceMapping(mesh, (Replicated(),))

        ep_batch_manager, comm_buffers = _build_ep_batch_manager(
            _ep_config(DType.bfloat16, num_devices), devices
        )

        with default_device(mesh), default_dtype(DType.bfloat16):
            layer = ExpertParallelMoE(
                hidden_dim=_HIDDEN_DIM,
                num_experts=_NUM_EXPERTS,
                num_experts_per_token=_NUM_EXPERTS_PER_TOKEN,
                moe_dim=_MOE_DIM,
                has_shared_experts=True,
                shared_experts_dim=_MOE_DIM,
                ep_batch_manager=ep_batch_manager,
            )

            x = Tensor.zeros(
                [_SEQ_LEN, _HIDDEN_DIM],
                dtype=DType.bfloat16,
                device=replicated,
            )
            out = layer(x, comm_buffers)
            ir = str(Graph.current._mlir_op)

    assert list(out.shape) == [_SEQ_LEN, _HIDDEN_DIM]
    assert "mo.distributed.ep" not in ir
    for op in ("dispatch_async", "dispatch_wait", "combine_async"):
        assert ir.count(f'"ep.{op}') == num_devices, op
    last_launch = ir.rindex('"ep.dispatch_async')
    first_wait = ir.index('"ep.dispatch_wait')
    # The shared expert's gate/up matmul is the only matmul in between: the
    # router runs before the launch and the routed experts after the wait.
    between = ir[last_launch:first_wait]
    assert "matmul" in between
    assert ('"mo.sequence"' in between) == (overlap_env == "1")


def test_expert_parallel_moe_fp8_weights(
    mock_accelerator: MagicMock, fp8_quant_config: QuantConfig
) -> None:
    """FP8 EP stacks per-device FP8 expert weights (data + scale)."""
    with F.lazy():
        devices = [mock_accelerator(0), mock_accelerator(1)]
        num_devices = len(devices)
        num_local_experts = _NUM_EXPERTS // num_devices
        mesh = DeviceMesh(tuple(devices), (num_devices,), (TP,))

        ep_batch_manager, _ = _build_ep_batch_manager(
            _ep_config(
                DType.float8_e4m3fn,
                num_devices,
                dispatch_quant_config=fp8_quant_config,
            ),
            devices,
        )

        with default_device(mesh):
            layer = ExpertParallelMoE(
                hidden_dim=_HIDDEN_DIM,
                num_experts=_NUM_EXPERTS,
                num_experts_per_token=_NUM_EXPERTS_PER_TOKEN,
                moe_dim=_MOE_DIM,
                quant_config=fp8_quant_config,
                ep_batch_manager=ep_batch_manager,
            )

        gate_up = layer.gate_up_proj
        assert len(gate_up) == num_devices
        for i, shard in enumerate(gate_up):
            assert isinstance(shard, FP8BlockTensor)
            assert list(shard.weight.shape) == [
                num_local_experts,
                2 * _MOE_DIM,
                _HIDDEN_DIM,
            ]
            assert shard.weight.device == devices[i]

        down = layer.down_proj
        assert len(down) == num_devices
        for shard in down:
            assert isinstance(shard, FP8BlockTensor)
            assert list(shard.weight.shape) == [
                num_local_experts,
                _HIDDEN_DIM,
                _MOE_DIM,
            ]


def test_tensor_parallel_moe_nvfp4_weights(
    mock_accelerator: MagicMock, nvfp4_quant_config: QuantConfig
) -> None:
    """NVFP4 TP co-shards the packed data and the 16-element block scales."""
    with F.lazy():
        devices = [mock_accelerator(0), mock_accelerator(1)]
        num_devices = len(devices)
        mesh = DeviceMesh(tuple(devices), (num_devices,), (TP,))

        with default_device(mesh):
            layer = TensorParallelMoE(
                hidden_dim=_HIDDEN_DIM,
                num_experts=_NUM_EXPERTS,
                num_experts_per_token=_NUM_EXPERTS_PER_TOKEN,
                moe_dim=_MOE_DIM,
                quant_config=nvfp4_quant_config,
            )

        gate_up = layer.gate_up_proj
        assert len(gate_up) == num_devices
        assert isinstance(gate_up[0], NVFP4Tensor)
        assert list(gate_up[0].weight.shape) == [
            _NUM_EXPERTS,
            2 * _MOE_DIM // num_devices,
            _HIDDEN_DIM // 2,
        ]
        assert list(gate_up[0].weight_scale.shape) == [
            _NUM_EXPERTS,
            2 * _MOE_DIM // num_devices,
            _HIDDEN_DIM // 16,
        ]
        # gate and up share one global scale, so the fused bundle keeps one
        # entry per expert even though it stacks 2 * num_experts weights.
        assert list(gate_up[0].weight_scale_2.shape) == [_NUM_EXPERTS]
        assert list(gate_up[0].input_scale.shape) == [_NUM_EXPERTS]

        down = layer.down_proj
        assert len(down) == num_devices
        assert isinstance(down[0], NVFP4Tensor)
        assert list(down[0].weight.shape) == [
            _NUM_EXPERTS,
            _HIDDEN_DIM,
            _MOE_DIM // 2 // num_devices,
        ]
        assert list(down[0].weight_scale.shape) == [
            _NUM_EXPERTS,
            _HIDDEN_DIM,
            _MOE_DIM // 16 // num_devices,
        ]
        # The per-tensor scales are replicated, not sharded.
        assert list(down[0].weight_scale_2.shape) == [_NUM_EXPERTS]
        assert list(down[0].input_scale.shape) == [_NUM_EXPERTS]


def test_tensor_parallel_moe_nvfp4_forward(
    mock_accelerator: MagicMock,
    nvfp4_quant_config: QuantConfig,
    sm100_arch: None,
) -> None:
    """NVFP4 TP forward returns a Partial sum for the caller to all-reduce."""
    with F.lazy():
        devices = [mock_accelerator(0), mock_accelerator(1)]
        mesh = DeviceMesh(tuple(devices), (len(devices),), (TP,))
        replicated = DeviceMapping(mesh, (Replicated(),))

        with default_device(mesh):
            layer = TensorParallelMoE(
                hidden_dim=_HIDDEN_DIM,
                num_experts=_NUM_EXPERTS,
                num_experts_per_token=_NUM_EXPERTS_PER_TOKEN,
                moe_dim=_MOE_DIM,
                quant_config=nvfp4_quant_config,
            )

        x = Tensor.zeros(
            [_SEQ_LEN, _HIDDEN_DIM],
            dtype=DType.bfloat16,
            device=replicated,
        )
        out = layer(x)

    assert list(out.shape) == [_SEQ_LEN, _HIDDEN_DIM]
    assert out.mapping.mesh == mesh
    assert out.mapping.placements == (Partial(),)


def test_expert_parallel_moe_nvfp4_weights(
    mock_accelerator: MagicMock, nvfp4_quant_config: QuantConfig
) -> None:
    """EP keeps whole NVFP4 experts per device, one global scale pair each."""
    with F.lazy():
        devices = [mock_accelerator(0), mock_accelerator(1)]
        num_devices = len(devices)
        num_local_experts = _NUM_EXPERTS // num_devices
        mesh = DeviceMesh(tuple(devices), (num_devices,), (TP,))

        ep_batch_manager, _ = _build_ep_batch_manager(
            _ep_config(
                DType.uint8,
                num_devices,
                dispatch_quant_config=nvfp4_quant_config,
            ),
            devices,
        )

        with default_device(mesh):
            layer = ExpertParallelMoE(
                hidden_dim=_HIDDEN_DIM,
                num_experts=_NUM_EXPERTS,
                num_experts_per_token=_NUM_EXPERTS_PER_TOKEN,
                moe_dim=_MOE_DIM,
                quant_config=nvfp4_quant_config,
                ep_batch_manager=ep_batch_manager,
            )

        gate_up = layer.gate_up_proj
        assert len(gate_up) == num_devices
        for i, shard in enumerate(gate_up):
            assert isinstance(shard, NVFP4Tensor)
            assert list(shard.weight.shape) == [
                num_local_experts,
                2 * _MOE_DIM,
                _HIDDEN_DIM // 2,
            ]
            assert list(shard.weight_scale.shape) == [
                num_local_experts,
                2 * _MOE_DIM,
                _HIDDEN_DIM // 16,
            ]
            # gate and up share one global scale per expert.
            assert list(shard.weight_scale_2.shape) == [num_local_experts]
            assert shard.weight.device == devices[i]

        down = layer.down_proj
        assert len(down) == num_devices
        for shard in down:
            assert isinstance(shard, NVFP4Tensor)
            assert list(shard.weight.shape) == [
                num_local_experts,
                _HIDDEN_DIM,
                _MOE_DIM // 2,
            ]
            assert list(shard.weight_scale_2.shape) == [num_local_experts]


def test_expert_parallel_moe_nvfp4_fused_swiglu_permutes_gate_up(
    mock_accelerator: MagicMock, nvfp4_quant_config: QuantConfig
) -> None:
    """The fused SwiGLU kernel gets row-interleaved gate/up weights.

    The permutation itself is shape-preserving (and value-checked in
    ``test_quant_ops``); what matters here is that the config flag reaches the
    weight property.
    """
    fused = dataclasses.replace(nvfp4_quant_config, can_use_fused_swiglu=True)
    with F.lazy():
        devices = [mock_accelerator(0), mock_accelerator(1)]
        num_devices = len(devices)
        num_local_experts = _NUM_EXPERTS // num_devices
        mesh = DeviceMesh(tuple(devices), (num_devices,), (TP,))

        def _build(
            quant_config: QuantConfig, swiglu_limit: float = 0.0
        ) -> ExpertParallelMoE:
            ep_batch_manager, _ = _build_ep_batch_manager(
                _ep_config(
                    DType.uint8,
                    num_devices,
                    dispatch_quant_config=quant_config,
                ),
                devices,
            )
            with default_device(mesh):
                return ExpertParallelMoE(
                    hidden_dim=_HIDDEN_DIM,
                    num_experts=_NUM_EXPERTS,
                    num_experts_per_token=_NUM_EXPERTS_PER_TOKEN,
                    moe_dim=_MOE_DIM,
                    quant_config=quant_config,
                    ep_batch_manager=ep_batch_manager,
                    swiglu_limit=swiglu_limit,
                )

        chained = _build(nvfp4_quant_config)
        assert not chained._uses_fused_swiglu

        # The fused kernel has no clamp, so a SwiGLU limit keeps the chained
        # path and its unpermuted weights.
        clamped = _build(fused, swiglu_limit=10.0)
        assert not clamped._uses_fused_swiglu

        layer = _build(fused)
        assert layer._uses_fused_swiglu

        gate_up = layer.gate_up_proj
        for shard in gate_up:
            assert isinstance(shard, NVFP4Tensor)
            assert list(shard.weight.shape) == [
                num_local_experts,
                2 * _MOE_DIM,
                _HIDDEN_DIM // 2,
            ]
            assert list(shard.weight_scale.shape) == [
                num_local_experts,
                2 * _MOE_DIM,
                _HIDDEN_DIM // 16,
            ]


# --------------------------------------------------------------------------- #
# GLM-5.x router
# --------------------------------------------------------------------------- #


def test_glm_router_outputs_keep_input_placement(
    mock_accelerator: MagicMock,
) -> None:
    """The DeepSeek-V3.2 router claims its top-k outputs' placement.

    ``moe_router_group_limited`` has no sharding rule, so its outputs are
    ``Unknown`` until claimed, and the expert-parallel MoE casts them next.
    """
    with F.lazy():
        devices = [mock_accelerator(0), mock_accelerator(1)]
        mesh = DeviceMesh(tuple(devices), (len(devices),), (TP,))
        replicated = DeviceMapping(mesh, (Replicated(),))

        with default_dtype(DType.bfloat16), default_device(mesh):
            router = DeepseekV3_2TopKRouter(
                hidden_dim=_HIDDEN_DIM,
                num_experts=_NUM_EXPERTS,
                num_experts_per_token=_NUM_EXPERTS_PER_TOKEN,
                routed_scaling_factor=1.0,
                scoring_func="sigmoid",
                topk_method="noaux_tc",
                n_group=2,
                topk_group=1,
                norm_topk_prob=True,
                correction_bias_dtype=DType.float32,
            )
            x = Tensor.zeros(
                [_SEQ_LEN, _HIDDEN_DIM],
                dtype=DType.bfloat16,
                device=replicated,
            )
            topk_idx, topk_weight = router(x)
            topk_idx = topk_idx.cast(DType.int32)

        assert topk_idx.mapping == replicated
        assert topk_weight.mapping == replicated


def _glm_moe_layer(
    mock_accelerator: MagicMock, quant_config: QuantConfig, dispatch: DType
) -> ExpertParallelMoE:
    """Builds the GLM-5.x expert-parallel MoE layer for ``quant_config``."""
    devices = [mock_accelerator(0), mock_accelerator(1)]
    mesh = DeviceMesh(tuple(devices), (len(devices),), (TP,))
    ep_batch_manager, _ = _build_ep_batch_manager(
        _ep_config(dispatch, len(devices), dispatch_quant_config=quant_config),
        devices,
    )
    config = SimpleNamespace(
        quant_config=quant_config,
        n_routed_experts=_NUM_EXPERTS,
        first_k_dense_replace=0,
        moe_layer_freq=1,
        routed_scaling_factor=1.0,
        scoring_func="sigmoid",
        topk_method="noaux_tc",
        n_group=2,
        topk_group=1,
        norm_topk_prob=True,
        correction_bias_dtype=DType.float32,
        hidden_size=_HIDDEN_DIM,
        num_experts_per_tok=_NUM_EXPERTS_PER_TOKEN,
        moe_intermediate_size=_MOE_DIM,
        n_shared_experts=1,
        mesh=mesh,
    )
    with default_dtype(DType.bfloat16), default_device(mesh):
        layer = _get_mlp(
            cast(DeepseekV3_2Config, config),
            ParallelismMode.TP_EP,
            0,
            ep_batch_manager,
        )
    assert isinstance(layer, ExpertParallelMoE)
    return layer


def test_glm_nvfp4_ep_router_and_shared_expert_in_bfloat16(
    mock_accelerator: MagicMock, nvfp4_quant_config: QuantConfig
) -> None:
    """NVFP4 EP matches the graph-API model's sharded router and shared expert.

    Those run the gate GEMM and the shared-expert SwiGLU in bfloat16; other
    quantizations keep the float32 variants.
    """
    with F.lazy():
        layer = _glm_moe_layer(
            mock_accelerator, nvfp4_quant_config, DType.uint8
        )
    assert not isinstance(layer.gate, DeepseekV3_2TopKRouter)
    assert type(layer.shared_experts) is QuantizedMLP


def test_glm_fp8_ep_router_and_shared_expert_in_float32(
    mock_accelerator: MagicMock, fp8_quant_config: QuantConfig
) -> None:
    with F.lazy():
        layer = _glm_moe_layer(
            mock_accelerator, fp8_quant_config, DType.float8_e4m3fn
        )
    assert isinstance(layer.gate, DeepseekV3_2TopKRouter)
    assert isinstance(layer.shared_experts, DeepseekV3_2MLP)


# --------------------------------------------------------------------------- #
# MXFP8
# --------------------------------------------------------------------------- #


def _mxfp8_tp_moe(
    devices: list[Device], quant_config: QuantConfig
) -> TensorParallelMoE:
    mesh = DeviceMesh(tuple(devices), (len(devices),), (TP,))
    with default_device(mesh):
        return TensorParallelMoE(
            hidden_dim=_HIDDEN_DIM,
            num_experts=_NUM_EXPERTS,
            num_experts_per_token=_NUM_EXPERTS_PER_TOKEN,
            moe_dim=_MOE_DIM,
            quant_config=quant_config,
        )


def test_tensor_parallel_moe_mxfp8_weights(
    mock_accelerator: MagicMock, mxfp8_quant_config: QuantConfig
) -> None:
    """MXFP8 TP co-shards the data and the 32-element block scales."""
    with F.lazy():
        devices = [mock_accelerator(0), mock_accelerator(1)]
        n = len(devices)
        layer = _mxfp8_tp_moe(devices, mxfp8_quant_config)

        gate_up = layer.gate_up_proj
        assert len(gate_up) == n
        assert isinstance(gate_up[0], MXFP8Tensor)
        assert list(gate_up[0].weight.shape) == [
            _NUM_EXPERTS,
            2 * _MOE_DIM // n,
            _HIDDEN_DIM,
        ]
        assert list(gate_up[0].weight_scale.shape) == [
            _NUM_EXPERTS,
            2 * _MOE_DIM // n,
            _HIDDEN_DIM // 32,
        ]

        down = layer.down_proj
        assert len(down) == n
        assert isinstance(down[0], MXFP8Tensor)
        assert list(down[0].weight.shape) == [
            _NUM_EXPERTS,
            _HIDDEN_DIM,
            _MOE_DIM // n,
        ]
        assert list(down[0].weight_scale.shape) == [
            _NUM_EXPERTS,
            _HIDDEN_DIM,
            _MOE_DIM // 32 // n,
        ]


def test_tensor_parallel_moe_mxfp8_forward(
    mock_accelerator: MagicMock,
    mxfp8_quant_config: QuantConfig,
    sm100_arch: None,
) -> None:
    """MXFP8 TP forward returns a Partial sum for the caller to all-reduce."""
    with F.lazy():
        devices = [mock_accelerator(0), mock_accelerator(1)]
        layer = _mxfp8_tp_moe(devices, mxfp8_quant_config)
        mesh = layer.mesh
        x = Tensor.zeros(
            [_SEQ_LEN, _HIDDEN_DIM],
            dtype=DType.bfloat16,
            device=DeviceMapping(mesh, (Replicated(),)),
        )
        out = layer(x)

    assert list(out.shape) == [_SEQ_LEN, _HIDDEN_DIM]
    assert out.mapping.placements == (Partial(),)


@pytest.mark.parametrize(
    ("quant_config_name", "moe_dim"),
    [
        # 176 columns per device: 5.5 MXFP8 blocks of 32.
        ("mxfp8_quant_config", 1408),
        # 24 columns per device: 1.5 NVFP4 blocks of 16.
        ("nvfp4_quant_config", 192),
    ],
)
def test_tensor_parallel_moe_rejects_split_scale_block(
    mock_accelerator: MagicMock,
    request: pytest.FixtureRequest,
    quant_config_name: str,
    moe_dim: int,
) -> None:
    """A TP slice of moe_dim that cuts a scale block would misalign the scales."""
    quant_config = request.getfixturevalue(quant_config_name)
    with F.lazy():
        devices = [mock_accelerator(i) for i in range(8)]
        mesh = DeviceMesh(tuple(devices), (len(devices),), (TP,))
        with (
            default_device(mesh),
            pytest.raises(ValueError, match="multiple of the weight scale"),
        ):
            TensorParallelMoE(
                hidden_dim=_HIDDEN_DIM,
                num_experts=_NUM_EXPERTS,
                num_experts_per_token=_NUM_EXPERTS_PER_TOKEN,
                moe_dim=moe_dim,
                quant_config=quant_config,
            )


def test_expert_parallel_moe_rejects_mxfp8(
    mock_accelerator: MagicMock, mxfp8_quant_config: QuantConfig
) -> None:
    """The EP dispatch has no MXFP8 path, so construction says so."""
    with F.lazy():
        devices = [mock_accelerator(0), mock_accelerator(1)]
        mesh = DeviceMesh(tuple(devices), (len(devices),), (TP,))
        ep_batch_manager, _ = _build_ep_batch_manager(
            _ep_config(DType.bfloat16, len(devices)), devices
        )
        with default_device(mesh), pytest.raises(ValueError, match="MXFP8"):
            ExpertParallelMoE(
                hidden_dim=_HIDDEN_DIM,
                num_experts=_NUM_EXPERTS,
                num_experts_per_token=_NUM_EXPERTS_PER_TOKEN,
                moe_dim=_MOE_DIM,
                quant_config=mxfp8_quant_config,
                ep_batch_manager=ep_batch_manager,
            )
