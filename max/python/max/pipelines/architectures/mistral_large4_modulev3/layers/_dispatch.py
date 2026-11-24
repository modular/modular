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
"""The target and shape checks the model's fused fp32 GEMV sites share.

Three places in this architecture ask the same question about a projection:
the router's gate, the indexer's head weights, and the decode-blocker
consultation that decides whether the row-invariant pins hold. They must agree
on what counts as SM100 and on which shapes keep an fp32 ``transpose_b``
matmul on the split-K GEMV, so both live here rather than beside each caller.
"""

from __future__ import annotations

from max.driver import accelerator_architecture_name, accelerator_count
from max.dtype import DType
from max.graph import StaticDim
from max.nn._graph_ops import _keeps_precise_fp32
from max.tensor import Tensor

SM100_ARCHS = frozenset(
    ("sm_100", "sm_100a", "sm_101", "sm_101a", "sm_103", "sm_103a")
)
"""The targets the SM100 matmul dispatch serves, as the stdlib's
``_has_blackwell_tcgen05`` names them."""


def on_sm100() -> bool:
    """Returns whether the graph being staged targets the SM100 family.

    Accelerator devices are required too, because the fused launches are
    GPU-only; a virtual-device staging run reports the arch it was told to
    stage for, which is what makes the same graphs compile off-box.
    """
    return (
        accelerator_count() > 0
        and accelerator_architecture_name() in SM100_ARCHS
    )


def fused_fp32_gemv_applies(x: Tensor, weight: Tensor) -> bool:
    """Returns whether a site can take the fused fp32 GEMV of its op.

    The activation must be a rank-2 BF16 tensor and the weight's ``[out,
    hidden]`` shape static, since the op specializes both from it; the shape
    must also satisfy the fp32 dispatch's own gate for keeping an fp32
    ``transpose_b`` matmul on the split-K GEMV, which is the launch every fused
    form reproduces; the graph must be on accelerator devices, because the ops
    are GPU-only; and the target must be SM100, the one family whose fp32 GEMV
    they reproduce bit for bit.

    Args:
        x: The ``[rows, hidden]`` activation the projection reads.
        weight: The ``[out, hidden]`` weight it projects with.
    """
    if len(x.shape) != 2 or x.dtype != DType.bfloat16:
        return False
    if len(weight.shape) != 2:
        return False
    n_dim, k_dim = weight.shape
    if not isinstance(n_dim, StaticDim) or not isinstance(k_dim, StaticDim):
        return False
    if not _keeps_precise_fp32(n_dim.dim, k_dim.dim):
        return False
    if all(device.is_host for device in x.mesh.devices):
        return False
    return on_sm100()
