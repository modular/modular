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

from collections import OptionalReg

from buffer.buffer import NDBuffer
from buffer.dimlist import Dim, DimList, _make_tuple
from gpu.host import DeviceContext
from random import rand
from linalg.grouped_matmul import grouped_matmul, naive_grouped_matmul
from linalg.utils import elementwise_epilogue_type
from linalg.utils_gpu import MatmulConfig
from testing import assert_almost_equal
from gpu.host.info import B200

from utils import IndexList
from utils.index import Index
import itertools
from layout import IntTuple, Layout, LayoutTensor
from layout._ndbuffer_stub import from_ndbuffer_row_major
from layout.runtime_layout import UNKNOWN_VALUE, RuntimeLayout


fn shrink_qkv_permute_3mn_sm100[
    c_type: DType,
    c_shape: DimList,
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList,
](
    c_lora: NDBuffer[c_type, 3, MutAnyOrigin, c_shape],
    a: NDBuffer[a_type, 2, ImmutAnyOrigin, a_shape],
    b: NDBuffer[b_type, 3, ImmutAnyOrigin, b_shape],
    a_offsets: NDBuffer[DType.uint32, 1, ImmutAnyOrigin],
    expert_ids: NDBuffer[DType.int32, 1, ImmutAnyOrigin],
    max_num_tokens_per_expert: Int,
    num_active_experts: Int,
    ctx: DeviceContext,
) raises:
    """LoRA shrink GMM with planar Q/K/V output on SM100.

    Performs the LoRA 'shrink' grouped matmul for routed tokens:
    computes `[M, K] @ [G, 3N, K]^T` per active expert, then **permutes**
    the flat `[M, 3N]` result into a planar layout `[3, M, N]` (Q, K, V)
    using an elementwise epilogue, while reusing the same storage.

    Args:
        c_lora: Output tensor with planar Q/K/V layout, shape (3, M, N).
                Backed by row-major storage, used both as a 3D view and as a
                temporary 2D view (M, 3N) during compute.
        a:      Routed activation matrix, shape (M, K).
        b:      Shrink weights per expert, shape (G, 3N, K).
        a_offsets: Inclusive prefix sums of tokens per (active) expert,
                length (num_experts + 1). Defines per-expert [start, end) in A/C.
        expert_ids: Expert indices for the active groups, length ≥ num_active_experts.
        max_num_tokens_per_expert: Upper bound on tokens for any active expert.
        num_active_experts: Number of experts participating in this call.
        ctx:    DeviceContext used for enqueues and synchronization.

    Constraints:
        - c_lora must be rank 3 with static first dimension B == 3.
        - a must be rank 2 with trailing dimension K that matches b[..., K].
        - b must be rank 3 with shape (G, 3N, K).
        - The temporary 2D view of c_lora is (M, 3N) in row-major order and
        **aliases the same storage** as c_lora.
        - a_offsets is non-decreasing with a_offsets[0] == 0 and
        a_offsets[num_active_experts] == M.
        - expert_ids[i] ∈ [0, G) for valid experts; kernel may treat -1 as inactive.
        - The epilogue assumes `N % vector_width == 0` for aligned vector stores.
    """
    var M = c_lora.dim[1]()
    var c_tensor_lora = from_ndbuffer_row_major(c_lora)  # LayoutTensor[3]
    comptime N = c_shape.get[2]()
    comptime B = c_shape.get[0]()
    comptime assert (
        c_shape.has_value[2]() and c_shape.get[0]() == 3
    ), "the outer dimension of c_shape must be known and equal to 3"
    comptime N_Total = B * N
    # Create an empty (null-backed) 2D NDBuffer for C with only shape/stride set.
    # This ensures GroupGEMM does NOT write into C directly; any changes to the
    # final C output must happen exclusively via the epilogue function.
    var c = NDBuffer[
        mut=True,
        c_type,
        2,
        MutAnyOrigin,
        shape = DimList(Dim(), Dim(N_Total)),
        strides = DimList.create_unknown[2](),
    ]()  # data=null, shape/stride zeroed

    # Populate the dynamic shape (row-major strides will be set later if needed).
    c.dynamic_shape = _make_tuple[2, element_type = DType.uint64](
        DimList(M, N_Total)
    )

    @always_inline
    @__copy_capture(c_tensor_lora, M)
    @parameter
    fn permute_dim_lora_bmn[
        dtype: DType, width: Int, *, alignment: Int = 1
    ](idx: IndexList[2], val: SIMD[dtype, width]) -> None:
        """Epilogue: permute flat (M, 3N) columns to planar (3, M, N) tiles.
        Maps a flat column index `j` into `(head, n)` via `divmod(j, N)` and
        stores the SIMD vector `val` into the original 3D layout tensor at
        `[head, m, n + lane]`. Used as the elementwise epilogue for the
        grouped matmul, so the final `c_lora` is written directly in
        planar Q/K/V format without an extra kernel.

        Args:
            idx: 2D index of the epilogue write in the temporary (M, 3N) view,
                where `idx[0] = m` and `idx[1] = j`.
            val: SIMD vector of partial results to be written for columns
                `j .. j+width-1` at row `m`.

        Constraints:
            - `N` is the per-head width; must satisfy `N % width == 0` for aligned
            and in-bounds vector stores.
            - The underlying storage of `c_tensor_lora` aliases the (M, 3N) view.
            - Rank/layout assumptions:
                * Input view is row-major (M, 3N).
                * Output view is row-major (3, M, N) with head-major tiles.
        """
        comptime N = c_shape.get[2]()
        var i = idx[0]
        var j = idx[1]
        var new_j, new_k = divmod(j, N)
        comptime assert N % width == 0, "N must be divisible by width"
        # The current index is [i, new_j, new_k] in the M x 3 x N row major
        # tensor.
        # The permdim tensor has the shape 3 x M x N, so the index is then
        # [new_j, i, new_k].
        var off = c_tensor_lora._offset(IndexList[3](new_j, i, new_k))
        c_tensor_lora.ptr.store[width=width, alignment=alignment](
            off, val.cast[c_type]()
        )

    # Run grouped_matmul and apply permute_dim_lora as the elementwise epilogue.
    grouped_matmul[elementwise_lambda_fn=permute_dim_lora_bmn,](
        c,
        a,
        b,
        a_offsets,
        expert_ids,
        max_num_tokens_per_expert,
        num_active_experts,
        ctx,
    )
