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
from max.gpu import thread_idx
from max.gpu.host import DeviceContext
from max.gpu.sync import barrier
from layout import row_major
from layout.swizzle import Swizzle
from layout.tile_io import LocalToSharedTileCopier
from layout.tile_tensor import stack_allocation
from std.testing import assert_equal


def copy_kernel[
    src_dtype: DType, dst_dtype: DType, M: Int, swizzled: Bool
](output: MutPointer[Scalar[dst_dtype], MutAnyOrigin],):
    comptime P = M // 8
    var local = stack_allocation[src_dtype, address_space=.LOCAL](
        row_major[P, 4]()
    )
    comptime for row in range(P):
        comptime for col in range(4):
            local[row, col] = (
                Scalar[src_dtype](
                    (thread_idx.x * 32 + row * 4 + col) % 127 - 63
                )
                / 8
            )
    var shared = stack_allocation[dst_dtype, address_space=.SHARED](
        row_major[M, 16]()
    )
    var fragment = local.vectorize[1, 2]().transpose()
    comptime if swizzled:
        LocalToSharedTileCopier[
            row_major[8, 4](), swizzle=Swizzle(3, 3, 1), num_threads=64
        ]().copy(shared.vectorize[1, 2](), fragment.as_imm())
    else:
        LocalToSharedTileCopier[row_major[8, 4](), num_threads=64]().copy(
            shared.vectorize[1, 2](), fragment.as_imm()
        )
    barrier()
    # Read the physical shared allocation without reversing the permutation.
    var tid = thread_idx.x
    var shared_flat = shared.reshape[M * 16]()
    for i in range(tid, M * 16, 64):
        output[i + 8] = shared_flat[i]


def test_copy[
    src_dtype: DType, dst_dtype: DType, M: Int, swizzled: Bool
](ctx: DeviceContext,) raises:
    comptime count = M * 16
    comptime P = M // 8
    var output = ctx.enqueue_create_buffer[dst_dtype](count + 16)
    with output.map_to_host() as host:
        for i in range(count + 16):
            host[i] = Scalar[dst_dtype](-99)
    ctx.enqueue_function[copy_kernel[src_dtype, dst_dtype, M, swizzled]](
        output, grid_dim=(1,), block_dim=(64,)
    )
    with output.map_to_host() as host:
        for i in range(8):
            assert_equal(host[i], Scalar[dst_dtype](-99))
            assert_equal(host[count + 8 + i], Scalar[dst_dtype](-99))
        for lane in range(32):
            for i in range(2 * P):
                var src_offset = (i // 2) * 4 + (i % 2) * 2
                var offset = (
                    (lane // 4 + (i % P) * 8) * 16
                    + (lane % 4) * 2
                    + (i // P) * 8
                )
                comptime if swizzled:
                    offset = offset ^ ((offset & 112) >> 1)
                for j in range(2):
                    var expected = (
                        Float32((lane * 32 + src_offset + j) % 127 - 63) / 8
                    )
                    assert_equal(
                        host[offset + j + 8], expected.cast[dst_dtype]()
                    )


def main() raises:
    with DeviceContext() as ctx:
        test_copy[.float32, .float16, 16, False](ctx)
        test_copy[.float32, .float16, 16, True](ctx)
        test_copy[.float32, .bfloat16, 16, False](ctx)
        test_copy[.float32, .bfloat16, 16, True](ctx)
        test_copy[.float32, .float16, 32, False](ctx)
        test_copy[.float32, .float16, 32, True](ctx)
        test_copy[.float32, .bfloat16, 32, False](ctx)
        test_copy[.float32, .bfloat16, 32, True](ctx)
        test_copy[.float32, .float32, 16, False](ctx)
        test_copy[.float32, .float32, 16, True](ctx)
        test_copy[.float16, .float16, 32, False](ctx)
        test_copy[.float16, .float16, 32, True](ctx)
