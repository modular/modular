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

"""Local-expert NVFP4 operations, preserving each projection's scalars."""

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.nn.common_layers import quant_ops
from max.nn.common_layers.functional_kernels import (
    moe_create_indices,
)
from max.nn.common_layers.quant_tensor import (
    NVFP4Tensor,
)


def swiglu(gate: Tensor, up: Tensor, limit: float | None = 10.0) -> Tensor:
    """Applies the asymmetric training clamp and SwiGLU in float32."""
    gate, up = gate.cast(DType.float32), up.cast(DType.float32)
    if limit is not None:
        gate = gate.clip(max=limit)
        up = up.clip(min=-limit, max=limit)
    return F.silu(gate) * up


def stack(weights: list[NVFP4Tensor]) -> NVFP4Tensor:
    """Stacks local weights and their per-expert scalars."""
    result = quant_ops.stack(list(weights))
    assert isinstance(result, NVFP4Tensor)
    return result


def routed(
    x: Tensor,
    ids: Tensor,
    weights: Tensor,
    w1: NVFP4Tensor,
    w3: NVFP4Tensor,
    w2: NVFP4Tensor,
    first: int,
    limit: float | None = 10.0,
) -> Tensor:
    """Computes one rank's float32 contribution from global expert ids.

    The grouped-matmul path: ``moe_create_indices`` builds the ragged
    expert groups, the three NVFP4 projections run as grouped block-scaled
    matmuls over the local stack, and the weighted rows are restored to
    token order and summed.
    """
    n = int(w1.weight.shape[0])
    topk = int(ids.shape[1])
    slots = ids.reshape([-1])
    is_local = (slots >= first) & (slots < first + n)
    local = F.where(is_local, slots - first, n).cast(DType.int32)
    # Invalid and nonlocal slots are initialized in a trailing group, which
    # no GEMM covers. Leaving ids out of range would leave restore unwritten.
    order, starts, restore, expert_ids, _, offsets = moe_create_indices(
        local, n + 1, needs_scales_offset=True
    )
    tokens = F.gather(x, (order // topk).cast(DType.int32), axis=0)
    starts, expert_ids, offsets = starts[: n + 1], expert_ids[:n], offsets[:n]
    gate = quant_ops.grouped_matmul(
        tokens, w1, starts, expert_ids, None, scales_offset=offsets
    )
    up = quant_ops.grouped_matmul(
        tokens, w3, starts, expert_ids, None, scales_offset=offsets
    )
    # Keep w1/w3 separate: concat_nvfp4 discards w3's scalar scales.
    activated = swiglu(gate, up, limit).cast(DType.bfloat16)
    down = quant_ops.grouped_matmul(
        activated, w2, starts, expert_ids, None, scales_offset=offsets
    )
    ordered = F.gather(down, restore.cast(DType.int32), axis=0).cast(
        DType.float32
    )
    ordered = F.where(is_local.reshape([-1, 1]), ordered, 0.0)
    # Weight AFTER GEMM2: weighting before its NVFP4 quantizer changes range.
    return (
        (
            ordered.reshape([x.shape[0], topk, x.shape[1]])
            * weights.reshape([x.shape[0], topk, 1])
        )
        .sum(axis=1)
        .squeeze(1)
    )
