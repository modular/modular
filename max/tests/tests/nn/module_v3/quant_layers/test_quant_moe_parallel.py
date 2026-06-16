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

from unittest.mock import MagicMock

import pytest
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn.common_layers.mesh_axis import TP
from max.experimental.sharding import (
    DeviceMesh,
    Partial,
    PlacementMapping,
    Replicated,
)
from max.experimental.tensor import Tensor, default_dtype
from max.nn.quant_config import QuantConfig
from max.pipelines.architectures.deepseekV3_modulev3.layers.quant_moe import (
    TensorParallelMoE,
)
from max.pipelines.architectures.deepseekV3_modulev3.layers.quant_tensor import (
    FP8BlockTensor,
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
# TensorParallelMoE
# --------------------------------------------------------------------------- #


def test_tensor_parallel_moe_bf16(mock_accelerator: MagicMock) -> None:
    """TP shards every expert's moe_dim; forward returns a Partial sum."""
    with F.lazy():
        devices = [mock_accelerator(0), mock_accelerator(1)]
        num_devices = len(devices)
        mesh = DeviceMesh(tuple(devices), (num_devices,), (TP,))
        replicated = PlacementMapping(mesh, (Replicated(),))

        with default_dtype(DType.bfloat16):
            layer = TensorParallelMoE(
                hidden_dim=_HIDDEN_DIM,
                num_experts=_NUM_EXPERTS,
                num_experts_per_token=_NUM_EXPERTS_PER_TOKEN,
                moe_dim=_MOE_DIM,
            ).to(mesh)

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
        assert out.mapping.to_placements() == (Partial(),)


def test_tensor_parallel_moe_bf16_shared_experts(
    mock_accelerator: MagicMock,
) -> None:
    """Routed + shared experts fold into one Partial sum (single all-reduce)."""
    with F.lazy():
        devices = [mock_accelerator(0), mock_accelerator(1)]
        num_devices = len(devices)
        mesh = DeviceMesh(tuple(devices), (num_devices,), (TP,))
        replicated = PlacementMapping(mesh, (Replicated(),))

        with default_dtype(DType.bfloat16):
            layer = TensorParallelMoE(
                hidden_dim=_HIDDEN_DIM,
                num_experts=_NUM_EXPERTS,
                num_experts_per_token=_NUM_EXPERTS_PER_TOKEN,
                moe_dim=_MOE_DIM,
                has_shared_experts=True,
                shared_experts_dim=_MOE_DIM,
            ).to(mesh)

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
        assert out.mapping.to_placements() == (Partial(),)


def test_tensor_parallel_moe_fp8_weights(
    mock_accelerator: MagicMock, fp8_quant_config: QuantConfig
) -> None:
    """FP8 TP shards both the packed data and the block-scale grid."""
    # TODO(MXF-296): Add distributed support for FP8 TP MoE.
    with F.lazy(), pytest.raises(AttributeError):
        devices = [mock_accelerator(0), mock_accelerator(1)]
        num_devices = len(devices)
        mesh = DeviceMesh(tuple(devices), (num_devices,), (TP,))

        layer = TensorParallelMoE(
            hidden_dim=_FP8_HIDDEN_DIM,
            num_experts=_NUM_EXPERTS,
            num_experts_per_token=_NUM_EXPERTS_PER_TOKEN,
            moe_dim=_FP8_MOE_DIM,
            quant_config=fp8_quant_config,
        ).to(mesh)

        gate_up = layer.gate_up_proj
        assert isinstance(gate_up, FP8BlockTensor)
        assert list(gate_up.data.shape) == [
            _NUM_EXPERTS,
            2 * _FP8_MOE_DIM // num_devices,
            _FP8_HIDDEN_DIM,
        ]
        # Scale grid is the (128, 128)-block count of the sharded data.
        assert list(gate_up.scale_inv.shape) == [
            _NUM_EXPERTS,
            2 * _FP8_MOE_DIM // num_devices // 128,
            _FP8_HIDDEN_DIM // 128,
        ]

        down = layer.down_proj
        assert isinstance(down, FP8BlockTensor)
        assert list(down.data.shape) == [
            _NUM_EXPERTS,
            _FP8_HIDDEN_DIM,
            _FP8_MOE_DIM // num_devices,
        ]
        assert list(down.scale_inv.shape) == [
            _NUM_EXPERTS,
            _FP8_HIDDEN_DIM // 128,
            _FP8_MOE_DIM // num_devices // 128,
        ]
