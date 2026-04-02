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

from std.sys import argv, size_of
from std.testing import assert_equal
import linalg.matmul.vendor.blas as vendor_blas
from std.gpu.host import DeviceContext
from std.gpu.host.nvidia.tma import TensorMapSwizzle
from std.memory import alloc

# from internal_utils import assert_almost_equal
from std.random import rand
from layout import (
    Coord,
    CoordLike,
    Idx,
    TileTensor,
    row_major,
)
from linalg.matmul.gpu.sm100_structured.default.matmul import (
    blackwell_batched_matmul_tma_umma_warp_specialized,
)
from linalg.matmul.gpu.sm100_structured.structured_kernels.config import (
    MatmulConfig,
)

from std.utils.index import Index, IndexList
from std.utils.static_tuple import StaticTuple


def simple_init() -> Bool:
    for arg in argv():
        if arg == "--simple-init":
            return True
    return False


def test_blackwell_batched_matmul_tma_umma_warp_specialized[
    BatchType: CoordLike,
    MType: CoordLike,
    NType: CoordLike,
    KType: CoordLike,
    //,
    a_type: DType,
    b_type: DType,
    c_type: DType,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    cluster_shape: StaticTuple[Int32, 3],
    cta_group: Int,
    transpose_b: Bool = True,
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    block_swizzle_size: Int = 0,
    swapAB: Bool = False,
    k_group_size: Int = 1,
    accum_dtype: DType = DType.float32,
](ctx: DeviceContext, batch: BatchType, m: MType, n: NType, k: KType) raises:
    var B = batch.value()
    var M = m.value()
    var N = n.value()
    var K = k.value()

    print(
        t"in/out dtypes=({a_type}, {b_type}, {c_type})  problem shape=({B},"
        t" {M}, {N}, {K})"
        t" mma_shape={mma_shape} block_tile_shape={block_tile_shape} cta_group={cta_group} cluster_shape=({cluster_shape[0]},"
        t" {cluster_shape[1]}, {cluster_shape[2]})"
        t" swapAB={swapAB} k_group_size={k_group_size}"
    )

    var a_shape = row_major(Coord(batch, m, Idx[KType.static_value]()))
    var b_shape = row_major(
        Coord(
            batch,
            Idx[NType.static_value if transpose_b else KType.static_value](),
            Idx[KType.static_value if transpose_b else NType.static_value](),
        )
    )
    var c_shape = row_major(Coord(batch, m, Idx[NType.static_value]()))
    var c_ref_shape = row_major(Coord(batch, m, Idx[NType.static_value]()))

    var a_size = batch.value() * m.value() * k.value()
    var b_size = batch.value() * n.value() * k.value()
    var c_size = batch.value() * m.value() * n.value()

    var a_host_ptr = alloc[Scalar[a_type]](a_size)
    var a_host = TileTensor(a_host_ptr, a_shape)
    var b_host_ptr = alloc[Scalar[b_type]](b_size)
    var b_host = TileTensor(b_host_ptr, b_shape)
    var c_host_ptr = alloc[Scalar[c_type]](c_size)
    var c_host = TileTensor(c_host_ptr, c_shape)
    var c_host_ref_ptr = alloc[Scalar[accum_dtype]](c_size)
    var c_host_ref = TileTensor(c_host_ref_ptr, c_ref_shape)

    var a_device = ctx.enqueue_create_buffer[a_type](a_size)
    var a_tensor = TileTensor(a_device.unsafe_ptr(), a_shape)
    var b_device = ctx.enqueue_create_buffer[b_type](b_size)
    var b_tensor = TileTensor(b_device.unsafe_ptr(), b_shape)
    var c_device = ctx.enqueue_create_buffer[c_type](c_size)
    var c_tensor = TileTensor(c_device.unsafe_ptr(), c_shape)
    var c_device_ref = ctx.enqueue_create_buffer[accum_dtype](c_size)
    var c_ref_tensor = TileTensor(c_device_ref.unsafe_ptr(), c_ref_shape)

    # Initialize matmul operands
    if simple_init():
        for b in range(B):
            for mi in range(M):
                for ki in range(K):
                    comptime assert a_host.flat_rank >= 3
                    a_host[(Idx(b), Idx(mi), Idx(ki))] = Float32(ki).cast[
                        a_type
                    ]()
        for b in range(B):
            for ni in range(N):
                for ki in range(K):
                    comptime assert b_host.flat_rank >= 3
                    b_host[(Idx(b), Idx(ni), Idx(ki))] = Float32(
                        1 if ni == ki else 0
                    ).cast[b_type]()
    else:
        rand(a_host.ptr, a_host.num_elements(), min=-1.0, max=1.0)
        rand(b_host.ptr, b_host.num_elements(), min=-1.0, max=1.0)

    # Move operands to device
    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)

    comptime matmul_config = MatmulConfig[a_type, b_type, c_type, transpose_b](
        cluster_shape=Index(
            cluster_shape[0], cluster_shape[1], cluster_shape[2]
        ),
        mma_shape=mma_shape,
        block_swizzle_size=block_swizzle_size,
        cta_group=cta_group,
        AB_swapped=swapAB,
        k_group_size=k_group_size,
    )

    blackwell_batched_matmul_tma_umma_warp_specialized[
        transpose_b=transpose_b,
        config=matmul_config,
    ](c_tensor, a_tensor, b_tensor, ctx)

    # Reference: per-batch vendor_blas.matmul
    for b in range(B):
        var a_2d = TileTensor(
            a_tensor.ptr + b * M * K,
            row_major((Idx(M), Idx(K))),
        )
        var b_2d = TileTensor(
            b_tensor.ptr + b * N * K,
            row_major((Idx(N), Idx(K))),
        )
        var c_ref_2d = TileTensor(
            c_ref_tensor.ptr + b * M * N,
            row_major((Idx(M), Idx(N))),
        )

        vendor_blas.matmul(
            ctx,
            c_ref_2d,
            a_2d,
            b_2d,
            c_row_major=True,
            transpose_b=transpose_b,
        )
        ctx.synchronize()

    ctx.enqueue_copy(c_host_ptr, c_device)
    ctx.enqueue_copy(c_host_ref_ptr, c_device_ref)
    ctx.synchronize()

    for b in range(B):
        for i in range(M):
            for j in range(N):
                comptime assert c_host.flat_rank >= 3
                assert_equal(
                    c_host[(Idx(b), Idx(i), Idx(j))].cast[DType.float64](),
                    c_host_ref[(Idx(b), Idx(i), Idx(j))]
                    .cast[c_type]()
                    .cast[DType.float64](),
                    msg="At [" + String(i) + ", " + String(j) + "]",
                )

    print("\n=== TEST PASSED ===\n")

    # Cleanup
    a_host_ptr.free()
    b_host_ptr.free()
    c_host_ptr.free()
    c_host_ref_ptr.free()
    _ = a_device^
    _ = b_device^
    _ = c_device^
    _ = c_device_ref^


def main() raises:
    with DeviceContext() as ctx:
        comptime dtype = DType.bfloat16
        comptime swizzle = TensorMapSwizzle.SWIZZLE_128B
        comptime BK = (swizzle.bytes() // size_of[dtype]())
        comptime MMA_K = 16
        comptime cta_group = 2

        # 256x256x16 MMA, 2x1x1 cluster
        comptime for bm in [128]:
            comptime for bn in [64, 128]:
                comptime block_tile_shape = Index(bm, bn, BK)
                comptime umma_shape = Index(
                    cta_group * bm, cta_group * bn, MMA_K
                )

                # Basic: small batch, small shape
                test_blackwell_batched_matmul_tma_umma_warp_specialized[
                    dtype,
                    dtype,
                    DType.float8_e4m3fn,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape=StaticTuple[Int32, 3](2, 1, 1),
                    cta_group=cta_group,
                ](
                    ctx,
                    Idx(Int(2)),
                    Idx(Int(128)),
                    Idx[128](),
                    Idx[128](),
                )

                # Medium
                test_blackwell_batched_matmul_tma_umma_warp_specialized[
                    dtype,
                    dtype,
                    DType.float8_e4m3fn,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape=StaticTuple[Int32, 3](2, 1, 1),
                    cta_group=cta_group,
                ](
                    ctx,
                    Idx(Int(4)),
                    Idx(Int(256)),
                    Idx[512](),
                    Idx[256](),
                )

                # Large, non-aligned M
                test_blackwell_batched_matmul_tma_umma_warp_specialized[
                    dtype,
                    dtype,
                    DType.float8_e4m3fn,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape=StaticTuple[Int32, 3](2, 1, 1),
                    cta_group=cta_group,
                    block_swizzle_size=8,
                ](
                    ctx,
                    Idx(Int(2)),
                    Idx(Int(1000)),
                    Idx[1024](),
                    Idx[1040](),
                )

                # Large batch
                test_blackwell_batched_matmul_tma_umma_warp_specialized[
                    dtype,
                    dtype,
                    DType.float8_e4m3fn,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape=StaticTuple[Int32, 3](4, 2, 1),
                    cta_group=cta_group,
                    block_swizzle_size=4,
                ](
                    ctx,
                    Idx(Int(16)),
                    Idx(Int(256)),
                    Idx[128](),
                    Idx[512](),
                )

                # Multi-cluster
                test_blackwell_batched_matmul_tma_umma_warp_specialized[
                    dtype,
                    dtype,
                    DType.float8_e4m3fn,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape=StaticTuple[Int32, 3](4, 4, 1),
                    cta_group=cta_group,
                    block_swizzle_size=1,
                ](
                    ctx,
                    Idx(Int(3)),
                    Idx(Int(500)),
                    Idx[2048](),
                    Idx[4096](),
                )

        # swapAB tests (2SM)
        comptime for bn in [64, 128]:
            comptime block_tile_shape = Index(128, bn, BK)
            comptime umma_shape = Index(cta_group * 128, cta_group * bn, MMA_K)

            test_blackwell_batched_matmul_tma_umma_warp_specialized[
                dtype,
                dtype,
                DType.float8_e4m3fn,
                block_tile_shape,
                umma_shape,
                cluster_shape=StaticTuple[Int32, 3](2, 1, 1),
                cta_group=cta_group,
                swapAB=True,
            ](
                ctx,
                Idx(Int(2)),
                Idx(Int(128)),
                Idx[128](),
                Idx[128](),
            )

            test_blackwell_batched_matmul_tma_umma_warp_specialized[
                dtype,
                dtype,
                DType.float8_e4m3fn,
                block_tile_shape,
                umma_shape,
                cluster_shape=StaticTuple[Int32, 3](2, 1, 1),
                cta_group=cta_group,
                swapAB=True,
                block_swizzle_size=4,
            ](
                ctx,
                Idx(Int(4)),
                Idx(Int(256)),
                Idx[512](),
                Idx[256](),
            )
