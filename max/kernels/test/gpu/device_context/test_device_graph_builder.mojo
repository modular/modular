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

from std.math import ceildiv
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.testing import assert_equal


def vec_add(
    output: UnsafePointer[Float32, MutAnyOrigin],
    in0: UnsafePointer[Float32, ImmutAnyOrigin],
    in1: UnsafePointer[Float32, ImmutAnyOrigin],
    length: Int,
):
    var tid = global_idx.x
    if tid >= length:
        return
    output[tid] = in0[tid] + in1[tid]


def test_vec_add_kernel_node(ctx: DeviceContext) raises:
    print("Test capturing and replaying a vec_add kernel in a device graph.")
    comptime length = 1024
    comptime block_dim = 256

    var in0_dev = ctx.enqueue_create_buffer[DType.float32](length)
    var in1_dev = ctx.enqueue_create_buffer[DType.float32](length)
    var out_dev = ctx.enqueue_create_buffer[DType.float32](length)

    with in0_dev.map_to_host() as in0_host, in1_dev.map_to_host() as in1_host:
        for i in range(length):
            in0_host[i] = Float32(i)
            in1_host[i] = Float32(length - i)

    var func = ctx.compile_function_experimental[vec_add]()
    var builder = ctx.create_graph_builder()
    builder.add_function(
        func,
        out_dev,
        in0_dev,
        in1_dev,
        length,
        grid_dim=ceildiv(length, block_dim),
        block_dim=block_dim,
    )
    var graph = builder^.instantiate()
    graph.replay()

    # Check values and zero out buffer for next run
    with out_dev.map_to_host() as out_host:
        for i in range(length):
            assert_equal(out_host[i], Float32(length))
            out_host[i] = 0.0

    graph.replay()

    with out_dev.map_to_host() as out_host:
        for i in range(length):
            assert_equal(out_host[i], Float32(length))


def test_closure_node(ctx: DeviceContext) raises:
    print("Test using a register_passable closure as a device graph node.")
    comptime length = 1024
    comptime block_dim = 256
    var scale = Float32(2.0)

    var in0_dev = ctx.enqueue_create_buffer[DType.float32](length)
    var in1_dev = ctx.enqueue_create_buffer[DType.float32](length)
    var out_dev = ctx.enqueue_create_buffer[DType.float32](length)

    with in0_dev.map_to_host() as in0_host, in1_dev.map_to_host() as in1_host:
        for i in range(length):
            in0_host[i] = Float32(i)
            in1_host[i] = Float32(length - i)

    var out_ptr = out_dev.unsafe_ptr()
    var in0_ptr = in0_dev.unsafe_ptr()
    var in1_ptr = in1_dev.unsafe_ptr()

    # Closure captures device pointers and scale from enclosing scope.
    def scaled_vec_add() register_passable {
        var scale, var out_ptr, var in0_ptr, var in1_ptr
    }:
        var tid = global_idx.x
        if tid >= length:
            return
        out_ptr[tid] = (in0_ptr[tid] + in1_ptr[tid]) * scale

    var builder = ctx.create_graph_builder()
    builder.add_function(
        scaled_vec_add,
        grid_dim=ceildiv(length, block_dim),
        block_dim=block_dim,
    )
    var graph = builder^.instantiate()
    graph.replay()

    with out_dev.map_to_host() as out_host:
        for i in range(length):
            assert_equal(out_host[i], Float32(length) * scale)


def main() raises:
    with DeviceContext() as ctx:
        test_vec_add_kernel_node(ctx)
        test_closure_node(ctx)
