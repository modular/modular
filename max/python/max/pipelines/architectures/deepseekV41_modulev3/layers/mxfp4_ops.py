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

"""Local-expert MXFP4 (W4A8) operations, as the reference computes them."""

from max import functional as F
from max.dtype import DType
from max.nn.common_layers import quant_ops
from max.nn.common_layers.quant_tensor import MXFP4Tensor
from max.nn.kernels import moe_create_indices
from max.tensor import Tensor

from .nvfp4_ops import swiglu


def routed(
    x: Tensor,
    ids: Tensor,
    weights: Tensor,
    w1: MXFP4Tensor,
    w3: MXFP4Tensor,
    w2: MXFP4Tensor,
    first: int,
    limit: float | None = 10.0,
) -> Tensor:
    """Computes one rank's float32 contribution from global expert ids.

    The reference's ``Expert.forward`` over grouped matmuls: each GEMM's
    input is quantized to MXFP8 (``act_quant(x, 32, "ue8m0")``), ``w1`` and
    ``w3`` run as one gate/up GEMM (MX weights carry no per-tensor scale,
    so stacking their rows changes no value), and the routing weight scales
    the SwiGLU output before the down projection quantizes it. The
    expert-parallel path (:meth:`.moe.RoutedExperts.dispatched`) applies it
    at the combine instead, after the down projection.
    """
    n = int(w1.weight.shape[0])
    inter = int(w1.weight.shape[1])
    topk = int(ids.shape[1])
    slots = ids.reshape([-1])
    is_local = (slots >= first) & (slots < first + n)
    local = F.where(is_local, slots - first, n).cast(DType.int32)
    # Invalid and nonlocal slots are initialized in a trailing group, which
    # no GEMM covers. Leaving ids out of range would leave restore unwritten.
    order, starts, restore, expert_ids, _, offsets = moe_create_indices(
        local, n + 1, needs_scales_offset=True
    )
    order = order.cast(DType.int32)
    tokens = F.gather(x, order // topk, axis=0)
    slot_weight = F.gather(weights.reshape([-1]), order, axis=0)
    starts, expert_ids, offsets = starts[: n + 1], expert_ids[:n], offsets[:n]
    gate_up = quant_ops.grouped_matmul(
        tokens,
        quant_ops.concat_weights(w1, w3, axis=1),
        starts,
        expert_ids,
        None,
        scales_offset=offsets,
    )
    gate, up = gate_up.split([inter, inter], axis=1)
    activated = swiglu(gate, up, limit) * slot_weight.unsqueeze(-1)
    down = quant_ops.grouped_matmul(
        activated.cast(DType.bfloat16),
        w2,
        starts,
        expert_ids,
        None,
        scales_offset=offsets,
    )
    ordered = F.gather(down, restore.cast(DType.int32), axis=0).cast(
        DType.float32
    )
    # ``where``, not a multiply: the masked rows hold whatever the GEMM
    # left there, NaN included.
    ordered = F.where(is_local.reshape([-1, 1]), ordered, 0.0)
    return (
        ordered.reshape([x.shape[0], topk, x.shape[1]]).sum(axis=1).squeeze(1)
    )
