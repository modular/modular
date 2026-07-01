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
"""Correctness test for the arch-agnostic naive block-scaled FP4 (NVFP4) GEMM.

Unlike the SM100 warp-specialized FP4 path, ``naive_block_scaled_matmul`` is
CUDA-core based and runs on any supported NVIDIA GPU, including consumer
Blackwell (sm_120 / sm_121). Uses a known-answer construction so no reference
matmul is required: every FP4 element decodes to 1.0 (packed byte 0x22) and
every block scale is 1.0, so with a unit per-tensor scale C[i, j] == K.
"""

from std.gpu.host import DeviceContext
from std.math import ceildiv
from std.testing import assert_equal
from layout import TileTensor, Coord, row_major, Idx
from linalg.fp4_quantization import naive_block_scaled_matmul
from linalg.fp4_utils import (
    NVFP4_SF_DTYPE,
    NVFP4_SF_VECTOR_SIZE,
    SF_MN_GROUP_SIZE,
    SF_ATOM_M,
    SF_ATOM_K,
)
from std.gpu.compute.arch.mma_nvidia_sm100 import UMMAKind


def test_naive_nvfp4_known_answer(ctx: DeviceContext) raises:
    comptime M = 256
    comptime N = 256
    comptime K = 256
    comptime a_type = DType.uint8  # two packed FP4-E2M1 per byte
    comptime c_type = DType.bfloat16
    comptime scales_dtype = NVFP4_SF_DTYPE  # float8_e4m3fn block scales
    comptime SF_VECTOR_SIZE = NVFP4_SF_VECTOR_SIZE

    var a_shape = row_major(Coord(Idx[M](), Idx[K // 2]()))
    var b_shape = row_major(Coord(Idx[N](), Idx[K // 2]()))
    var c_shape = row_major(Coord(Idx[M](), Idx[N]()))

    var a_dev = ctx.enqueue_create_buffer[a_type](M * (K // 2))
    var b_dev = ctx.enqueue_create_buffer[a_type](N * (K // 2))
    var c_dev = ctx.enqueue_create_buffer[c_type](M * N)
    var a = TileTensor(a_dev, a_shape)
    var b = TileTensor(b_dev, b_shape)
    var c = TileTensor(c_dev, c_shape)

    var a_host_ptr = ctx.enqueue_create_host_buffer[a_type](M * (K // 2))
    var b_host_ptr = ctx.enqueue_create_host_buffer[a_type](N * (K // 2))
    var c_host_ptr = ctx.enqueue_create_host_buffer[c_type](M * N)
    var a_host = TileTensor(a_host_ptr, a_shape)
    var b_host = TileTensor(b_host_ptr, b_shape)
    var c_host = TileTensor(c_host_ptr, c_shape)

    # Byte 0x22 = two e2m1 values of 1.0 each.
    for i in range(a_host.num_elements()):
        a_host.ptr[i] = UInt8(0x22)
    for i in range(b_host.num_elements()):
        b_host.ptr[i] = UInt8(0x22)

    var a_sc_shape = row_major(
        Coord(
            Idx[ceildiv(M, SF_MN_GROUP_SIZE)](),
            Idx[ceildiv(K, SF_VECTOR_SIZE * SF_ATOM_K)](),
            Idx[SF_ATOM_M[0]](),
            Idx[SF_ATOM_M[1]](),
            Idx[SF_ATOM_K](),
        )
    )
    var b_sc_shape = row_major(
        Coord(
            Idx[ceildiv(N, SF_MN_GROUP_SIZE)](),
            Idx[ceildiv(K, SF_VECTOR_SIZE * SF_ATOM_K)](),
            Idx[SF_ATOM_M[0]](),
            Idx[SF_ATOM_M[1]](),
            Idx[SF_ATOM_K](),
        )
    )
    var a_sc_total = a_sc_shape.product()
    var b_sc_total = b_sc_shape.product()

    var a_sc_host_ptr = ctx.enqueue_create_host_buffer[scales_dtype](a_sc_total)
    var b_sc_host_ptr = ctx.enqueue_create_host_buffer[scales_dtype](b_sc_total)
    var a_sc_host = TileTensor(a_sc_host_ptr, a_sc_shape)
    var b_sc_host = TileTensor(b_sc_host_ptr, b_sc_shape)
    for i in range(a_sc_host.num_elements()):
        a_sc_host.ptr[i] = Scalar[scales_dtype](1.0)
    for i in range(b_sc_host.num_elements()):
        b_sc_host.ptr[i] = Scalar[scales_dtype](1.0)

    var a_sc_dev = ctx.enqueue_create_buffer[scales_dtype](a_sc_total)
    var b_sc_dev = ctx.enqueue_create_buffer[scales_dtype](b_sc_total)
    var a_sc = TileTensor(a_sc_dev, a_sc_shape)
    var b_sc = TileTensor(b_sc_dev, b_sc_shape)

    ctx.enqueue_copy(a_dev, a_host_ptr)
    ctx.enqueue_copy(b_dev, b_host_ptr)
    ctx.enqueue_copy(a_sc_dev, a_sc_host_ptr)
    ctx.enqueue_copy(b_sc_dev, b_sc_host_ptr)

    naive_block_scaled_matmul[
        scaling_kind=UMMAKind.KIND_MXF4NVF4,
        SF_VECTOR_SIZE=SF_VECTOR_SIZE,
        transpose_b=True,
    ](
        c.to_layout_tensor(),
        a.to_layout_tensor(),
        b.to_layout_tensor(),
        a_sc.to_layout_tensor(),
        b_sc.to_layout_tensor(),
        ctx,
        1.0,
    )
    ctx.synchronize()
    ctx.enqueue_copy(c_host_ptr, c_dev)
    ctx.synchronize()

    # All-1.0 inputs with unit scales => every output element equals K.
    var bad = 0
    for i in range(c_host.num_elements()):
        if abs(Float32(c_host.ptr[i]) - Float32(K)) > 0.5:
            bad += 1
    assert_equal(bad, 0)

    _ = a_dev^
    _ = b_dev^
    _ = c_dev^
    _ = a_sc_dev^
    _ = b_sc_dev^


def main() raises:
    with DeviceContext() as ctx:
        test_naive_nvfp4_known_answer(ctx)
