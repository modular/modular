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

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.testing import assert_equal

comptime width = 8


def run_insert[
    dtype: DType, input_width: Int, offset: Int
](ctx: DeviceContext) raises:
    var output_host = ctx.enqueue_create_host_buffer[dtype](width)
    var output_buffer = ctx.enqueue_create_buffer[dtype](width)
    output_buffer.enqueue_fill(0)

    def kernel(output: MutPointer[Scalar[dtype], MutAnyOrigin]):
        if global_idx.x != 0:
            return
        var vec = SIMD[dtype, width]()
        comptime for i in range(width):
            vec[i] = Scalar[dtype](i)
        var sub = SIMD[dtype, input_width](Scalar[dtype](100))
        output.unsafe_store(0, vec.insert[offset=offset](sub))

    ctx.enqueue_function[kernel](output_buffer, grid_dim=1, block_dim=1)
    ctx.enqueue_copy(output_host, output_buffer)
    ctx.synchronize()

    for i in range(width):
        if offset <= i < offset + input_width:
            assert_equal(output_host[i], Scalar[dtype](100))
        else:
            assert_equal(output_host[i], Scalar[dtype](i))


def main() raises:
    with DeviceContext() as ctx:
        run_insert[.float32, 1, 3](ctx)
        run_insert[.int32, 1, 0](ctx)
        run_insert[.float32, 2, 4](ctx)
        run_insert[.int32, 4, 4](ctx)
