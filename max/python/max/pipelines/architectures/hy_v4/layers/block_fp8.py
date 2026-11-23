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
"""Hy4's block-FP8 linears with power-of-two (UE8M0) scales.

vLLM serves Hy4-FP8's linears with DeepGEMM's UE8M0 recipe on Blackwell: it
re-quantizes each 128x128 weight block to a power-of-two scale at load
(``fp8_utils.py`` ``requant_weight_ue8m0_inplace``) and rounds each
activation scale up to a power of two. The checkpoint's fp32 scales give
different tokens, so the linears here do the same.
"""

from max import functional as F
from max.dtype import DType
from max.nn.common_layers import quant_ops
from max.nn.common_layers.quant_tensor import FP8BlockTensor
from max.tensor import Tensor

_FP8_MAX = 448.0
# vLLM's ``per_block_cast_to_fp8`` floors a weight block's amax here.
_AMAX_FLOOR = 1e-4


def _requant_ue8m0(weight: FP8BlockTensor) -> FP8BlockTensor:
    """Returns ``weight`` re-quantized to power-of-two block scales.

    Each block decodes to float32 and takes the scale
    ``2**ceil(log2(max(amax, 1e-4) / 448))``, as vLLM's
    ``per_block_cast_to_fp8(use_ue8m0=True)`` does.
    """
    rows, cols = (int(d) for d in weight.weight.shape)
    block_m, block_k = weight.block_size
    blocks = weight.dequantize(DType.float32).reshape(
        [rows // block_m, block_m, cols // block_k, block_k]
    )
    amax = F.max(F.max(F.max(F.abs(blocks), axis=3), axis=1), _AMAX_FLOOR)
    # The E8M0 cast rounds up, so it is the ceil to a power of two.
    scales = (amax / _FP8_MAX).cast(DType.float8_e8m0fnu).cast(DType.float32)
    codes = (blocks * (1.0 / scales)).cast(DType.float8_e4m3fn)
    return FP8BlockTensor(
        weight=codes.reshape([rows, cols]),
        weight_scale_inv=scales.reshape([rows // block_m, cols // block_k]),
        block_size=weight.block_size,
    )


def matmul(x: Tensor, weight: FP8BlockTensor) -> Tensor:
    """Returns ``x @ weight.T`` on UE8M0 weight and activation scales."""
    return quant_ops.matmul(x, _requant_ue8m0(weight), pow2_scales=True)
