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

from std.gpu.host import DeviceContext, FuncAttribute
from layout import ComptimeInt, RowMajorLayout
from linalg.matmul.gpu._multistage_gemm_gpu import multistage_gemm_kernel
from linalg.utils_gpu import MatmulKernels


def multistage_gemm_simple[
    M: Int,
    N: Int,
    K: Int,
    a_type: DType = DType.bfloat16,
    b_type: DType = DType.bfloat16,
    c_type: DType = DType.bfloat16,
    transpose_b: Bool = False,
](ctx: DeviceContext,) raises:
    comptime kernels = MatmulKernels[a_type, b_type, c_type, transpose_b]()
    comptime config = kernels.ampere_128x128_4

    comptime a_tt_layout = RowMajorLayout[ComptimeInt[M], ComptimeInt[K]]
    # Select the b dims first (an `Int` ternary unifies cleanly); a ternary
    # over the two `RowMajorLayout[...]` types directly does not, since the
    # transposed/non-transposed layouts are distinct parameterized types.
    comptime b_dim0 = N if transpose_b else K
    comptime b_dim1 = K if transpose_b else N
    comptime b_tt_layout = RowMajorLayout[
        ComptimeInt[b_dim0], ComptimeInt[b_dim1]
    ]
    comptime c_tt_layout = RowMajorLayout[ComptimeInt[M], ComptimeInt[N]]

    # Dispatch w/o split K
    comptime gemm_kernel_type = multistage_gemm_kernel[
        c_type,
        c_tt_layout,
        a_type,
        a_tt_layout,
        b_type,
        b_tt_layout,
        transpose_b,
        c_linear_idx_type=DType.int64,
        a_linear_idx_type=DType.int64,
        b_linear_idx_type=DType.int64,
        config=config,
    ]

    var gemm_kernel = ctx.compile_function[gemm_kernel_type](
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(config.shared_mem_usage())
        ),
    )


def main() raises:
    with DeviceContext() as ctx:
        multistage_gemm_simple[
            1024,
            1024,
            1024,
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            False,
        ](ctx)
        multistage_gemm_simple[
            1024,
            1024,
            1024,
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            False,
        ](ctx)

        multistage_gemm_simple[
            550, 2048, 8, DType.float32, DType.float32, DType.float32, False
        ](ctx)
