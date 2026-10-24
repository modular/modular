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
"""The MoE SwiGLU clamp in ``quant_ops.grouped_silu``, against PyTorch.

With a non-zero ``swiglu_limit`` the activation is
``silu(min(gate, limit)) * clamp(up, -limit, limit)``. Both the BF16 path
(the fused SiLU kernel) and the FP8 path (SiLU fused into the block quantize)
must apply it. The inputs reach well past the limit, so an unclamped kernel
fails the comparison.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from max.driver import CPU, Accelerator
from max.dtype import DType
from max.experimental.nn.common_layers import quant_ops
from max.experimental.nn.common_layers.quant_tensor import (
    FP8BlockActivation,
    FP8BlockTensor,
)
from max.experimental.tensor import Tensor

_ROWS = 64
_MOE_DIM = 256
_BLOCK_K = 128
_LIMIT = 10.0


def _gate_up(seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    # Up to 3x the limit on both sides, so most rows have clamped entries.
    return (torch.rand(_ROWS, 2 * _MOE_DIM, generator=gen) * 6 - 3) * _LIMIT


def _reference(x: torch.Tensor, limit: float) -> torch.Tensor:
    gate, up = x[:, :_MOE_DIM].double(), x[:, _MOE_DIM:].double()
    if limit:
        gate = gate.clamp(max=limit)
        up = up.clamp(-limit, limit)
    return torch.nn.functional.silu(gate) * up


def _inputs(x: torch.Tensor) -> tuple[Tensor, Tensor]:
    device = Accelerator()
    offsets = np.array([0, _ROWS], dtype=np.uint32)
    return (
        Tensor(x.to(torch.bfloat16).cuda(), device=device),
        Tensor.from_dlpack(offsets).to(device),
    )


@pytest.mark.parametrize("limit", [0.0, _LIMIT])
def test_bf16_silu_applies_the_limit(limit: float) -> None:
    x = _gate_up(1).to(torch.bfloat16).float()
    got = quant_ops.grouped_silu(*_inputs(x), swiglu_limit=limit)
    assert isinstance(got, Tensor)
    got_t = torch.from_dlpack(got.to(CPU())).double()

    want = _reference(x, limit)
    torch.testing.assert_close(got_t, want, rtol=2e-2, atol=2e-2)
    if limit:
        assert not torch.allclose(got_t, _reference(x, 0.0), rtol=0.1)


def test_fp8_silu_quantize_applies_the_limit() -> None:
    x = _gate_up(2).to(torch.bfloat16).float()
    device = Accelerator()
    # Only the down weight's block size is read: it sets the re-quant's K
    # block.
    down = FP8BlockTensor(
        weight=Tensor.zeros(
            [_MOE_DIM, _MOE_DIM], dtype=DType.float8_e4m3fn, device=device
        ),
        weight_scale_inv=Tensor.ones(
            [_MOE_DIM // _BLOCK_K, _MOE_DIM // _BLOCK_K],
            dtype=DType.float32,
            device=device,
        ),
        block_size=(_BLOCK_K, _BLOCK_K),
    )
    got = quant_ops.grouped_silu(*_inputs(x), down, swiglu_limit=_LIMIT)
    assert isinstance(got, FP8BlockActivation)

    data = torch.from_dlpack(got.data.cast(DType.float32).to(CPU()))
    scales = torch.from_dlpack(got.scales.to(CPU()))
    k_blocks = _MOE_DIM // _BLOCK_K
    if tuple(scales.shape) != (_ROWS, k_blocks):
        scales = scales.T
    assert tuple(scales.shape) == (_ROWS, k_blocks)
    dequant = data.double() * scales.double().repeat_interleave(_BLOCK_K, 1)

    # One FP8 e4m3 step is 1/8 relative at the top of a block's range.
    want = _reference(x, _LIMIT)
    torch.testing.assert_close(dequant, want, rtol=0.13, atol=0.13 * _LIMIT)
    assert not torch.allclose(dequant, _reference(x, 0.0), rtol=0.13)
