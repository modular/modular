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
"""Tests for max.experimental.nn.norm.group_norm."""

from __future__ import annotations

import pytest
import torch
from max.driver import Accelerator, accelerator_count
from max.dtype import DType
from max.experimental.nn.norm import GroupNorm
from max.experimental.nn.norm.group_norm import group_norm
from max.experimental.tensor import Tensor


def test_repr() -> None:
    assert "num_groups=4" in repr(GroupNorm(4, 8))
    assert "num_channels=8" in repr(GroupNorm(4, 8))


def test_invalid_init() -> None:
    with pytest.raises(ValueError, match="divisible"):
        GroupNorm(5, 11)


def test_parameters_affine() -> None:
    norm = GroupNorm(4, 8, affine=True)
    params = dict(norm.parameters)
    assert "weight" in params
    assert "bias" in params


def test_parameters_no_affine() -> None:
    norm = GroupNorm(4, 8, affine=False)
    params = dict(norm.parameters)
    assert len(params) == 0


@pytest.mark.skipif(not accelerator_count(), reason="requires accelerator")
@pytest.mark.parametrize(
    "num_channels,num_groups,affine",
    [
        (4, 2, True),
        (8, 4, True),
        (6, 3, False),
    ],
)
def test_group_norm_call(
    num_channels: int, num_groups: int, affine: bool
) -> None:
    """Verify the experimental GroupNorm compiles and runs on GPU."""
    device = Accelerator()
    norm = GroupNorm(num_groups, num_channels, affine=affine).to(device)
    x = Tensor.ones([2, num_channels, 4], device=device)
    result = norm(x)
    assert result.shape == [2, num_channels, 4]


@pytest.mark.skipif(not accelerator_count(), reason="requires accelerator")
def test_group_norm_functional() -> None:
    """Verify the functional group_norm API works end-to-end."""
    device = Accelerator()
    x = Tensor.ones([1, 4, 8], device=device)
    weight = Tensor.ones([4], device=device)
    bias = Tensor.zeros([4], device=device)
    result = group_norm(x, weight, bias, num_groups=2, epsilon=1e-5)
    assert result.shape == [1, 4, 8]


# When the graph compiler view-fuses a transpose (NHWC→NCHW, perm [0,3,1,2])
# directly into group_norm, the generated strided_load uses stride=C (the
# physical stride of the W-dimension in NHWC memory).  This is correct within
# a single c_offset region of the flattened group but wrong when a simd_width
# load straddles a c_offset boundary (at a multiple of H*W).  The straddling
# can only happen when H*W % simd_width != 0.
#
# The fix in group_norm_gpu's input_fn_2d detects the boundary and falls back
# to element-wise scalar loads (width=1 per element) for that one thread.
@pytest.mark.skipif(not accelerator_count(), reason="requires accelerator")
@pytest.mark.parametrize(
    "N,C,H,W,num_groups",
    [
        # H*W not divisible by simd_width(bf16)=8 — exercises the fix path.
        (1, 128, 7, 9, 32),  # H*W = 63
        (1, 64, 3, 5, 16),  # H*W = 15
        (2, 128, 7, 7, 32),  # H*W = 49
        # H*W divisible by 8 — fast path, regression guard.
        (1, 128, 8, 8, 32),  # H*W = 64
        (1, 128, 4, 16, 32),  # H*W = 64
    ],
)
def test_group_norm_view_fused_transpose(
    N: int, C: int, H: int, W: int, num_groups: int
) -> None:
    """GEX-3544: group_norm on a view-fused NHWC→NCHW permute matches ref.

    The GC fuses the permute as a view into group_norm.  For shapes where
    H*W % simd_width != 0, the kernel's vectorized load can straddle a
    c_offset boundary and read wrong data from the strided NHWC buffer.
    """
    torch.manual_seed(42)
    device = Accelerator()
    eps = 1e-6

    # Random NHWC input in bfloat16 (simd_width=8 on GPU for bf16).
    x_nhwc_torch = torch.randn(N, H, W, C, dtype=torch.bfloat16)
    # Contiguous NCHW reference for PyTorch.
    x_nchw_torch = x_nhwc_torch.permute(0, 3, 1, 2).contiguous()

    # PyTorch reference: unit weight and bias (default for our GroupNorm).
    ref_layer = torch.nn.GroupNorm(num_groups, C, eps=eps).bfloat16()
    torch.nn.init.ones_(ref_layer.weight)
    torch.nn.init.zeros_(ref_layer.bias)
    ref_out = ref_layer(x_nchw_torch).detach().float()

    # MAX: permute inside the graph so the GC can view-fuse it into group_norm.
    x_dev = Tensor.from_dlpack(x_nhwc_torch.cuda())
    weight_dev = Tensor.ones([C], dtype=DType.bfloat16, device=device)
    bias_dev = Tensor.zeros([C], dtype=DType.bfloat16, device=device)

    x_dev_nchw = x_dev.permute([0, 3, 1, 2])
    out = group_norm(
        x_dev_nchw, weight_dev, bias_dev, num_groups=num_groups, epsilon=eps
    )
    max_out = torch.from_dlpack(out).cpu().float()

    torch.testing.assert_close(max_out, ref_out, rtol=0.02, atol=0.02)
