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
"""Unified NVFP4/MXFP8 grouped block-scaled matmul + SwiGLU dispatch."""

from std.gpu.host import DeviceContext
from std.gpu.compute.arch.mma_nvidia_sm100 import UMMAKind
from std.gpu.primitives.grid_controls import PDLLevel
from layout import TileTensor

from .dispatch import _scaling_kind
from .grouped_matmul_swiglu_nvfp4 import grouped_matmul_swiglu_nvfp4_dispatch
from .grouped_matmul_swiglu_mxfp8 import grouped_matmul_swiglu_mxfp8_dispatch


def grouped_matmul_block_scaled_swiglu_sm100_dispatch[
    transpose_b: Bool = True,
    target: StaticString = "cpu",
    pdl_level: PDLLevel = PDLLevel.ON,
    clamp_activation: Bool = False,
](
    c: TileTensor,
    c_swiglu_scales: TileTensor,
    a: TileTensor,
    b: TileTensor,
    a_scales: TileTensor,
    b_scales: TileTensor,
    a_offsets: TileTensor,
    a_scale_offsets: TileTensor,
    expert_ids: TileTensor,
    expert_scales: TileTensor,
    c_input_scales: TileTensor,
    num_active_experts: Int,
    estimated_total_m: Int,
    ctx: DeviceContext,
    alpha: Float32 = Float32(0.0),
    limit: Float32 = Float32(0.0),
) raises:
    """Dispatches grouped block-scaled matmul with fused SwiGLU by dtype."""

    comptime scaling_kind = _scaling_kind[a.dtype, a_scales.dtype]()

    comptime if scaling_kind == UMMAKind.KIND_MXF4NVF4:
        grouped_matmul_swiglu_nvfp4_dispatch[
            transpose_b=transpose_b,
            target=target,
            pdl_level=pdl_level,
            clamp_activation=clamp_activation,
        ](
            c,
            c_swiglu_scales,
            a,
            b,
            a_scales,
            b_scales,
            a_offsets,
            a_scale_offsets,
            expert_ids,
            expert_scales,
            c_input_scales,
            num_active_experts,
            estimated_total_m,
            ctx,
            alpha,
            limit,
        )
    elif scaling_kind == UMMAKind.KIND_MXF8F6F4:
        grouped_matmul_swiglu_mxfp8_dispatch[
            transpose_b=transpose_b,
            target=target,
            pdl_level=pdl_level,
            clamp_activation=clamp_activation,
        ](
            c,
            c_swiglu_scales,
            a,
            b,
            a_scales,
            b_scales,
            a_offsets,
            a_scale_offsets,
            expert_ids,
            expert_scales,
            num_active_experts,
            estimated_total_m,
            ctx,
            alpha,
            limit,
        )
    else:
        raise Error(t"Unsupported scaling kind: {scaling_kind}")
