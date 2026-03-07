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
# mojo build --debug-level=full --mcmodel=medium --large-data-threshold=1048576
# to build this file if running into linking issues with large PTX kernels.

from std.random import random_si64
from std.math import ceildiv

from buffer import Dim, DimList, NDBuffer
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from layout._utils import ManagedLayoutTensor
from layout.tile_tensor import TileTensor
from layout.tile_layout import row_major
from layout.coord import Coord, Idx
from linalg.matmul.gpu import _matmul_gpu, matmul_kernel_naive
from std.utils import IndexList

comptime epilogue_func_type = fn[
    type: DType, width: Int, *, alignment: Int = 1
](IndexList[2], IndexList[2], SIMD[type, width]) capturing -> SIMD[type, width]

comptime to_dim[value: Optional[Int]] = value.value() if value else Dim()


@parameter
@always_inline
fn epilogue_test_fn[
    dtype: DType, width: Int, *, alignment: Int = 1
](
    idx: IndexList[2],
    dim_space: IndexList[2],
    val: SIMD[dtype, width],
) -> SIMD[
    dtype, width
]:
    var bias = SIMD[dtype, width](0)

    comptime for i in range(width):
        bias[i] = (
            0.5
            + Float64(idx[0] + idx[1] + i)
            / Float64(dim_space[0] + dim_space[1])
        ).cast[dtype]()

    return val + bias


fn test[
    in_type: DType,
    out_type: DType,
    transpose_b: Bool,
    M: Optional[Int],
    N: Optional[Int],
    K: Optional[Int],
](ctx: DeviceContext, m: Int, n: Int, k: Int,) raises:
    comptime assert Bool(N) and Bool(
        K
    ), "This test currently requires static N and K."

    print(m, "x", n, "x", k, "transpose_b", transpose_b)

    comptime static_a_shape = DimList(to_dim[M], to_dim[K])
    comptime static_b_shape = DimList(
        to_dim[N], to_dim[K]
    ) if transpose_b else DimList(to_dim[K], to_dim[N])
    comptime static_c_shape = DimList(to_dim[M], to_dim[N])

    var dynamic_a_shape = IndexList[2](M.or_else(m), K.or_else(k))
    var dynamic_b_shape = IndexList[2](
        N.or_else(n), K.or_else(k)
    ) if transpose_b else IndexList[2](K.or_else(k), N.or_else(n))
    var dynamic_c_shape = IndexList[2](M.or_else(m), N.or_else(n))

    var a_size = m * k
    var b_size = n * k if transpose_b else k * n
    var c_size = m * n

    comptime a_layout = Layout.row_major(
        M.or_else(UNKNOWN_VALUE), K.or_else(UNKNOWN_VALUE)
    )
    comptime b_layout = Layout.row_major(
        N.or_else(UNKNOWN_VALUE), K.or_else(UNKNOWN_VALUE)
    ) if transpose_b else Layout.row_major(
        K.or_else(UNKNOWN_VALUE), N.or_else(UNKNOWN_VALUE)
    )
    comptime c_layout = Layout.row_major(
        M.or_else(UNKNOWN_VALUE), N.or_else(UNKNOWN_VALUE)
    )

    var a_managed = ManagedLayoutTensor[in_type, a_layout](
        RuntimeLayout[a_layout].row_major(dynamic_a_shape),
        ctx,
    )
    var b_managed = ManagedLayoutTensor[in_type, b_layout](ctx)
    var c_managed = ManagedLayoutTensor[out_type, c_layout](
        RuntimeLayout[c_layout].row_major(dynamic_c_shape),
        ctx,
    )
    var c_ref_managed = ManagedLayoutTensor[out_type, c_layout](
        RuntimeLayout[c_layout].row_major(dynamic_c_shape),
        ctx,
    )

    var a_host = a_managed.tensor[update=False]()
    var b_host = b_managed.tensor[update=False]()
    var c_host = c_managed.tensor[update=False]()
    var c_host_ref = c_ref_managed.tensor[update=False]()

    comptime rand_min = -100
    comptime rand_max = 100

    for i in range(m * k):
        var val = random_si64(rand_min, rand_max)
        a_host.ptr[i] = val.cast[in_type]()

    for i in range(k * n):
        var val = random_si64(rand_min, rand_max)
        b_host.ptr[i] = val.cast[in_type]()

    for i in range(m * n):
        c_host.ptr[i] = 0
        c_host_ref.ptr[i] = 0

    var a_device_tensor = a_managed.device_tensor()
    var b_device_tensor = b_managed.device_tensor()
    var c_device_tensor = c_managed.device_tensor()
    var c_device_ref_tensor = c_ref_managed.device_tensor()

    var a_device = NDBuffer[in_type, 2, _, static_a_shape](
        a_device_tensor.ptr,
        IndexList[2](m, k),
    )
    var b_device = NDBuffer[in_type, 2, _, static_b_shape](
        b_device_tensor.ptr,
        IndexList[2](n, k) if transpose_b else IndexList[2](k, n),
    )
    var c_device = NDBuffer[out_type, 2, _, static_c_shape](
        c_device_tensor.ptr,
        IndexList[2](m, n),
    )
    var c_device_ref = NDBuffer[out_type, 2, _, static_c_shape](
        c_device_ref_tensor.ptr,
        IndexList[2](m, n),
    )

    _matmul_gpu[use_tensor_core=True, transpose_b=transpose_b](
        c_device,
        a_device,
        b_device,
        ctx,
    )

    var c_host_final = c_managed.tensor()

    comptime BLOCK_DIM = 16
    var c_ref_tt = TileTensor(
        c_device_ref_tensor.ptr,
        row_major(Coord(Idx(m), Idx(n))),
    )
    var a_tt = TileTensor(
        UnsafePointer[Scalar[in_type], ImmutAnyOrigin](
            unsafe_from_address=Int(a_device_tensor.ptr)
        ),
        row_major(Coord(Idx(dynamic_a_shape[0]), Idx(dynamic_a_shape[1]))),
    )
    var b_tt = TileTensor(
        UnsafePointer[Scalar[in_type], ImmutAnyOrigin](
            unsafe_from_address=Int(b_device_tensor.ptr)
        ),
        row_major(Coord(Idx(dynamic_b_shape[0]), Idx(dynamic_b_shape[1]))),
    )
    comptime naive_kernel = matmul_kernel_naive[
        out_type,
        in_type,
        in_type,
        type_of(c_ref_tt).LayoutType,
        type_of(a_tt).LayoutType,
        type_of(b_tt).LayoutType,
        BLOCK_DIM,
        transpose_b=transpose_b,
    ]
    ctx.enqueue_function_experimental[naive_kernel](
        c_ref_tt,
        a_tt,
        b_tt,
        m,
        n,
        k,
        grid_dim=(ceildiv(m, BLOCK_DIM), ceildiv(n, BLOCK_DIM)),
        block_dim=(BLOCK_DIM, BLOCK_DIM),
    )
    ctx.synchronize()

    var c_host_ref_final = c_ref_managed.tensor()

    var errors = 0
    for i in range(m * n):
        if c_host_final.ptr[i] != c_host_ref_final.ptr[i]:
            errors += 1

    print("errors", errors)
    if errors > 0:
        raise "GEMV failed: exact mismatch"


fn test_gemv_split_k[in_type: DType](ctx: DeviceContext) raises:
    """Test GEMV_SPLIT_K path: M=1, transpose_b=True, K % simd_width == 0."""
    print("=== Testing GEMV_SPLIT_K with", in_type, "===")

    comptime out_type = DType.float32 if in_type == DType.bfloat16 else DType.bfloat16

    # Basic GEMV_SPLIT_K
    test[
        in_type=in_type,
        out_type=out_type,
        transpose_b=True,
        M=None,
        N=Int(4096),
        K=Int(4096),
    ](ctx, 1, 4096, 4096)

    # N % TILE_N != 0 (bounds checking)
    test[
        in_type=in_type,
        out_type=out_type,
        transpose_b=True,
        M=None,
        N=Int(75837),
        K=Int(5120),
    ](ctx, 1, 75837, 5120)

    # Large K (tests 512-thread / 256-thread dispatch + unroll)
    test[
        in_type=in_type,
        out_type=out_type,
        transpose_b=True,
        M=None,
        N=Int(16384),
        K=Int(16384),
    ](ctx, 1, 16384, 16384)

    # Small K (tests 64-thread dispatch)
    test[
        in_type=in_type,
        out_type=out_type,
        transpose_b=True,
        M=None,
        N=Int(16384),
        K=Int(1024),
    ](ctx, 1, 16384, 1024)

    # Mid K (tests 128-thread dispatch)
    test[
        in_type=in_type,
        out_type=out_type,
        transpose_b=True,
        M=None,
        N=Int(16384),
        K=Int(2048),
    ](ctx, 1, 16384, 2048)


def main() raises:
    with DeviceContext() as ctx:
        # GEMV_SPLIT_K tests for bf16 and fp8
        test_gemv_split_k[DType.bfloat16](ctx)
        test_gemv_split_k[DType.float8_e4m3fn](ctx)

        # BF16-only tests for other GEMV paths (FP8 not supported here)

        # GEMV_KERNEL_VECTOR: N = 1, K % simd_width == 0, transpose_b = False
        test[
            in_type=DType.bfloat16,
            out_type=DType.float32,
            transpose_b=False,
            M=None,
            N=Int(1),
            K=Int(4096),
        ](ctx, 4096, 1, 4096)

        # GEMV_KERNEL_VECTOR: N = 1, K % simd_width == 0, transpose_b = True
        test[
            in_type=DType.bfloat16,
            out_type=DType.bfloat16,
            transpose_b=True,
            M=None,
            N=Int(1),
            K=Int(13824),
        ](ctx, 5120, 1, 13824)

        # GEMV_KERNEL: M = 1, K % simd_width !=0, transpose_b = True
        test[
            in_type=DType.bfloat16,
            out_type=DType.float32,
            transpose_b=True,
            M=None,
            N=Int(4096),
            K=Int(4095),
        ](ctx, 1, 4096, 4095)

        # GEMV_KERNEL: N = 1, K % simd_width !=0, transpose_b = False
        test[
            in_type=DType.bfloat16,
            out_type=DType.float32,
            transpose_b=False,
            M=None,
            N=Int(1),
            K=Int(4095),
        ](ctx, 4096, 1, 4095)

        # matmul_naive: M = 1, K % WARP_SIZE != 0, transpose_b = False
        test[
            in_type=DType.bfloat16,
            out_type=DType.float32,
            transpose_b=False,
            M=None,
            N=Int(4096),
            K=Int(4095),
        ](ctx, 1, 4096, 4095)
