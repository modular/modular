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
"""`QuantizedMoE` routes its FP8 experts without a host sync on SM100.

The non-expert-parallel MoE derives the FP8 grouped matmul's host usage stats
from shapes instead of copying the device-resident ones from
`moe_create_indices`. On SM100 that must give the result the host copy gives,
with no device-to-host copy enqueued, and the copy must remain elsewhere.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from max import driver
from max.driver import Accelerator, LaunchTraceEntry
from max.driver.driver import accelerator_architecture_name
from max.dtype import DType
from max.nn.common_layers.quant_moe import QuantizedMoE
from max.nn.common_layers.quant_tensor import FP8BlockTensor
from max.nn.legacy import kernels
from max.nn.legacy.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)
from max.tensor import Tensor

_E = 4
_TOP_K = 2
_HIDDEN = 256
_MOE_DIM = 128
_BLOCK = 128
# 8 tokens x top-2 = 16 routed rows; 7 tokens = 14, which is not a multiple of
# the quantize op's 4-row scale padding.
_TOKENS = [8, 7]


pytestmark = pytest.mark.skipif(
    not driver.accelerator_count()
    or not accelerator_architecture_name().startswith("sm_10"),
    reason="requires an SM100-class (Blackwell datacenter) GPU",
)


def _fp8_config() -> QuantConfig:
    return QuantConfig(
        input_scale=InputScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            origin=ScaleOrigin.DYNAMIC,
            dtype=DType.float32,
            block_size=(1, _BLOCK),
        ),
        weight_scale=WeightScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            dtype=DType.float32,
            block_size=(_BLOCK, _BLOCK),
        ),
        mlp_quantized_layers=set(),
        attn_quantized_layers=set(),
        embedding_output_dtype=None,
        format=QuantFormat.BLOCKSCALED_FP8,
    )


def _stacked(
    generator: torch.Generator, device: Accelerator, n: int, k: int
) -> FP8BlockTensor:
    weight = (torch.randn(_E, n, k, generator=generator) * 0.2).to(
        torch.float8_e4m3fn
    )
    scale = (
        torch.rand(_E, n // _BLOCK, k // _BLOCK, generator=generator) * 0.02
        + 0.005
    )
    return FP8BlockTensor(
        weight=Tensor.from_dlpack(weight.cuda()),
        weight_scale_inv=Tensor.from_dlpack(scale.cuda()),
        block_size=(_BLOCK, _BLOCK),
    )


@pytest.mark.parametrize("tokens", _TOKENS)
def test_routed_experts_match_the_host_copy_without_a_d2h_copy(
    tokens: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    device = Accelerator()
    generator = torch.Generator().manual_seed(5)
    moe = QuantizedMoE(
        hidden_dim=_HIDDEN,
        num_experts=_E,
        num_experts_per_token=_TOP_K,
        moe_dim=_MOE_DIM,
        quant_config=_fp8_config(),
    )
    gate_up = _stacked(generator, device, 2 * _MOE_DIM, _HIDDEN)
    down = _stacked(generator, device, _HIDDEN, _MOE_DIM)
    x = Tensor.from_dlpack(
        torch.randn(tokens, _HIDDEN, generator=generator)
        .to(torch.bfloat16)
        .cuda()
    )
    router_idx = Tensor.from_dlpack(
        (np.arange(tokens * _TOP_K, dtype=np.int32) % _E).reshape(
            tokens, _TOP_K
        )
    ).to(device)
    router_weight = Tensor.from_dlpack(
        torch.rand(tokens, _TOP_K, generator=generator).cuda()
    )

    def run() -> tuple[torch.Tensor, list[str]]:
        args = (x, router_idx, router_weight, gate_up, down)
        moe._routed_experts(*args)
        with driver.launch_trace() as entries:
            out = moe._routed_experts(*args)
            assert out.real
        d2h = [
            e.name
            for e in entries
            if e.kind == LaunchTraceEntry.OperationKind.MEMCPY
            and e.memcpy_kind == LaunchTraceEntry.MemcpyKind.DTOH
        ]
        return torch.from_dlpack(out).cpu().clone(), d2h

    got, copies = run()
    assert copies == []

    # The control: without the SM100 path the stats are copied back.
    monkeypatch.setattr(kernels, "_is_sm10x_gpu", lambda: False)
    want, control_copies = run()
    assert control_copies != []
    assert torch.equal(got, want)
