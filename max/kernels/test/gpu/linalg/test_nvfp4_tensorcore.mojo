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
"""Consumer-Blackwell native NVFP4 tensor-core GEMM test.

Known-answer construction: every FP4 nibble decodes to 1.0 and every e4m3 scale
is 1.0, so C[i, j] == K. Exercises the mma.sync.mxf4nvf4 warp path reading the
rank-5 SF layout across several shapes. Requires a sm_120a/sm_121a GPU on a
CUDA 13.1+ driver (PTX 9.1); tagged `manual` (see BUILD.bazel).
"""

from std.gpu.host import DeviceContext
from std.math import ceildiv
from std.testing import assert_equal
from layout import TileTensor, row_major, Idx
from linalg.nvfp4_tensorcore import nvfp4_tensorcore_block_scaled_matmul
from linalg.fp4_utils import NVFP4_SF_DTYPE, SF_ATOM_M, SF_ATOM_K


def _run(ctx: DeviceContext, M: Int, N: Int, K: Int) raises:
    var KB = K // 2
    var G_K = ceildiv(K, 64)
    var sfa = ceildiv(M, 128) * G_K * 32 * 4 * 4
    var sfb = ceildiv(N, 128) * G_K * 32 * 4 * 4

    var a_buf = ctx.enqueue_create_buffer[DType.uint8](M * KB)
    var b_buf = ctx.enqueue_create_buffer[DType.uint8](N * KB)
    var asc_buf = ctx.enqueue_create_buffer[NVFP4_SF_DTYPE](sfa)
    var bsc_buf = ctx.enqueue_create_buffer[NVFP4_SF_DTYPE](sfb)
    var c_buf = ctx.enqueue_create_buffer[DType.float32](M * N)

    var a_h = ctx.enqueue_create_host_buffer[DType.uint8](M * KB)
    var b_h = ctx.enqueue_create_host_buffer[DType.uint8](N * KB)
    var asc_h = ctx.enqueue_create_host_buffer[NVFP4_SF_DTYPE](sfa)
    var bsc_h = ctx.enqueue_create_host_buffer[NVFP4_SF_DTYPE](sfb)
    var c_h = ctx.enqueue_create_host_buffer[DType.float32](M * N)
    # all FP4 nibbles = 0x2 (1.0), all e4m3 scales = 1.0  =>  C == K
    for i in range(M * KB):
        a_h[i] = UInt8(0x22)
    for i in range(N * KB):
        b_h[i] = UInt8(0x22)
    for i in range(sfa):
        asc_h[i] = Scalar[NVFP4_SF_DTYPE](1.0)
    for i in range(sfb):
        bsc_h[i] = Scalar[NVFP4_SF_DTYPE](1.0)
    ctx.enqueue_copy(a_buf, a_h)
    ctx.enqueue_copy(b_buf, b_h)
    ctx.enqueue_copy(asc_buf, asc_h)
    ctx.enqueue_copy(bsc_buf, bsc_h)

    var a = TileTensor(a_buf, row_major(M, KB))
    var b = TileTensor(b_buf, row_major(N, KB))
    var c = TileTensor(c_buf, row_major(M, N))
    var asc = TileTensor(
        asc_buf, row_major((ceildiv(M, 128), G_K, Idx[32], Idx[4], Idx[4]))
    )
    var bsc = TileTensor(
        bsc_buf, row_major((ceildiv(N, 128), G_K, Idx[32], Idx[4], Idx[4]))
    )

    nvfp4_tensorcore_block_scaled_matmul(
        c.to_layout_tensor(),
        a.to_layout_tensor(),
        b.to_layout_tensor(),
        asc.to_layout_tensor(),
        bsc.to_layout_tensor(),
        ctx,
    )
    ctx.synchronize()
    ctx.enqueue_copy(c_h, c_buf)
    ctx.synchronize()

    var bad = 0
    for i in range(M * N):
        if abs(Float32(c_h[i]) - Float32(K)) > 0.5:
            bad += 1
    assert_equal(bad, 0)

    _ = a_buf^
    _ = b_buf^
    _ = asc_buf^
    _ = bsc_buf^
    _ = c_buf^


def test_nvfp4_tensorcore_known_answer(ctx: DeviceContext) raises:
    _run(ctx, 64, 64, 256)
    _run(ctx, 512, 512, 256)
    _run(ctx, 256, 128, 192)
    _run(ctx, 2048, 2048, 256)


def main() raises:
    with DeviceContext() as ctx:
        test_nvfp4_tensorcore_known_answer(ctx)
