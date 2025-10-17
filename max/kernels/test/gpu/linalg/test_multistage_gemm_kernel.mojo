# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from gpu.host import DeviceContext, FuncAttribute
from layout import Layout
from linalg.matmul.gpu._multistage_gemm_gpu import multistage_gemm_kernel
from linalg.utils_gpu import MatmulKernels


fn multistage_gemm_simple[
    M: Int,
    N: Int,
    K: Int,
    a_type: DType = DType.bfloat16,
    b_type: DType = DType.bfloat16,
    c_type: DType = DType.bfloat16,
    transpose_b: Bool = False,
](ctx: DeviceContext,) raises:
    alias kernels = MatmulKernels[a_type, b_type, c_type, transpose_b]()
    alias config = kernels.ampere_128x128_4

    alias a_layout = Layout.row_major(M, K)
    alias b_layout = Layout.row_major(
        N, K
    ) if transpose_b else Layout.row_major(K, N)
    alias c_layout = Layout.row_major(M, N)

    # Dispatch w/o split K
    alias gemm_kernel_type = multistage_gemm_kernel[
        c_type,
        c_layout,
        a_type,
        a_layout,
        b_type,
        b_layout,
        transpose_b,
        c_layout_int_type = DType.int64,
        c_linear_idx_type = DType.int64,
        a_layout_int_type = DType.int64,
        a_linear_idx_type = DType.int64,
        b_layout_int_type = DType.int64,
        b_linear_idx_type = DType.int64,
        config=config,
    ]

    var gemm_kernel = ctx.compile_function[gemm_kernel_type](
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            config.shared_mem_usage()
        ),
    )


def main():
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
