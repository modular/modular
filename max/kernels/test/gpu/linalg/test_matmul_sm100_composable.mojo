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

from hashlib import default_comp_time_hasher
from math import ceildiv
from sys import size_of

import linalg.matmul.vendor.blas as vendor_blas
from buffer.buffer import NDBuffer
from buffer.dimlist import DimList
from gpu import WARP_SIZE, barrier
from gpu.cluster import block_rank_in_cluster
from gpu.host import DeviceContext, FuncAttribute
from gpu.host.nvidia.tma import TensorMapSwizzle
from gpu import block_idx, lane_id, thread_idx
from gpu.memory import external_memory
from gpu.mma_sm100 import *
from gpu.tcgen05 import *
from memory import LegacyUnsafePointer as UnsafePointer

# Additional imports for testing
from internal_utils import assert_almost_equal, random, zero
from internal_utils._utils import ValOrDim, dynamic, static
from layout import Layout, LayoutTensor
from layout._ndbuffer_stub import from_ndbuffer_row_major
from layout.int_tuple import IntTuple
from layout.tensor_core_async import tile_layout_k_major, tile_layout_mn_major
from layout.tma_async import SharedMemBarrier, TMATensorTile, create_tma_tile
from linalg.arch.sm100 import MmaOpSM100_SS
from linalg.matmul.gpu.sm100.composable import matmul_sm100

from utils.index import Index, IndexList
from utils.numerics import get_accum_type
from utils.static_tuple import StaticTuple


def test_blackwell_matmul_tma_umma[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    umma_shape: IndexList[3],
    transpose_b: Bool = True,
    BK: Int = 64,
    benchmark: Bool = False,
](ctx: DeviceContext, m: ValOrDim, n: ValOrDim, k: ValOrDim):
    var M = m.value
    var N = n.value
    var K = k.value

    if not benchmark:
        print(M, "x", N, "x", K)

    comptime static_a_shape = DimList(m.dim, k.dim)
    comptime static_b_shape = DimList(n.dim, k.dim) if transpose_b else DimList(
        k.dim, n.dim
    )
    comptime static_c_shape = DimList(m.dim, n.dim)
    var dynamic_a_shape = DimList(m.value, k.value)
    var dynamic_b_shape = DimList(n.value, k.value) if transpose_b else DimList(
        k.value, n.value
    )
    var dynamic_c_shape = DimList(m.value, n.value)

    var a_size = m.value * k.value
    var b_size = n.value * k.value if transpose_b else k.value * n.value
    var c_size = m.value * n.value

    var a_host_ptr = UnsafePointer[Scalar[a_type]].alloc(a_size)
    var a_host = NDBuffer[a_type, 2, _, static_a_shape](
        a_host_ptr, dynamic_a_shape
    )
    var b_host_ptr = UnsafePointer[Scalar[b_type]].alloc(b_size)
    var b_host = NDBuffer[b_type, 2, _, static_b_shape](
        b_host_ptr, dynamic_b_shape
    )
    var c_host_ptr = UnsafePointer[Scalar[c_type]].alloc(c_size)
    var c_host = NDBuffer[c_type, 2, _, static_c_shape](
        c_host_ptr, dynamic_c_shape
    )
    var c_host_ref_ptr = UnsafePointer[Scalar[c_type]].alloc(c_size)
    var c_host_ref = NDBuffer[c_type, 2, _, static_c_shape](
        c_host_ref_ptr, dynamic_c_shape
    )

    var a_device = ctx.enqueue_create_buffer[a_type](a_size)
    var a_device_nd = NDBuffer[a_type, 2, _, static_a_shape](
        a_device.unsafe_ptr(), dynamic_a_shape
    )
    var b_device = ctx.enqueue_create_buffer[b_type](b_size)
    var b_device_nd = NDBuffer[b_type, 2, _, static_b_shape](
        b_device.unsafe_ptr(), dynamic_b_shape
    )
    var c_device = ctx.enqueue_create_buffer[c_type](c_size)
    var c_device_nd = NDBuffer[c_type, 2, _, static_c_shape](
        c_device.unsafe_ptr(), dynamic_c_shape
    )
    var c_device_ref = ctx.enqueue_create_buffer[c_type](c_size)
    var c_device_ref_nd = NDBuffer[c_type, 2, _, static_c_shape](
        c_device_ref.unsafe_ptr(), dynamic_c_shape
    )

    # Initialize matmul operands
    random(a_host)
    random(b_host)
    zero(c_host)
    zero(c_host_ref)

    # Move operands to the Device

    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)

    ctx.enqueue_copy(c_device, c_host_ptr)
    ctx.enqueue_copy(c_device_ref, c_host_ref_ptr)

    var a = from_ndbuffer_row_major(a_device_nd)
    var b = from_ndbuffer_row_major(b_device_nd)
    var c = from_ndbuffer_row_major(c_device_nd)

    comptime block_tile_shape = Index(umma_shape[0], umma_shape[1], BK)

    matmul_sm100[mma_shape=umma_shape, block_tile_shape=block_tile_shape](
        c_device_nd, a_device_nd, b_device_nd, ctx
    )

    ctx.synchronize()

    vendor_blas.matmul(
        ctx,
        c_device_ref_nd,
        a_device_nd,
        b_device_nd,
        c_row_major=True,
        transpose_b=transpose_b,
    )

    ctx.synchronize()

    ctx.enqueue_copy(c_host_ptr, c_device)
    ctx.enqueue_copy(c_host_ref_ptr, c_device_ref)
    ctx.synchronize()
    comptime rtol = 1e-2
    assert_almost_equal(
        c_host,
        c_host_ref,
        atol=0.0001,
        rtol=rtol,
    )

    print("passed")

    # Cleanup
    a_host_ptr.free()
    b_host_ptr.free()
    c_host_ptr.free()
    c_host_ref_ptr.free()
    _ = a_device^
    _ = b_device^
    _ = c_device^
    _ = c_device_ref^


def main():
    with DeviceContext() as ctx:
        test_blackwell_matmul_tma_umma[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            umma_shape = Index(64, 128, 16),
            transpose_b=True,
            BK=64,
        ](ctx, dynamic(128), static[128](), static[128]())

        test_blackwell_matmul_tma_umma[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            umma_shape = Index(64, 128, 16),
            transpose_b=True,
            BK=64,
        ](ctx, dynamic(1024), static[2048](), static[2048]())

        comptime BK_list: List[Int] = [64, 128]

        @parameter
        for BK in BK_list:
            test_blackwell_matmul_tma_umma[
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                umma_shape = Index(64, 128, 16),
                transpose_b=True,
                BK=BK,
            ](ctx, dynamic(1024), static[2048](), static[2048]())

            test_blackwell_matmul_tma_umma[
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                umma_shape = Index(64, 128, 16),
                transpose_b=True,
                BK=BK,
            ](ctx, static[1024](), static[2048](), static[2048]())

            test_blackwell_matmul_tma_umma[
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                umma_shape = Index(64, 128, 16),
                transpose_b=True,
                BK=BK,
            ](ctx, dynamic(100), static[512](), static[256]())

            test_blackwell_matmul_tma_umma[
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                umma_shape = Index(64, 128, 16),
                transpose_b=True,
                BK=BK,
            ](ctx, dynamic(99), static[1024](), static[1024]())

            test_blackwell_matmul_tma_umma[
                DType.bfloat16,
                DType.bfloat16,
                DType.bfloat16,
                umma_shape = Index(64, 128, 16),
                transpose_b=True,
                BK=BK,
            ](ctx, dynamic(201), static[2048](), static[256]())
