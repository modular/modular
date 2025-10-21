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

from gpu import *
from gpu.host import DeviceContext
from testing import assert_equal, TestSuite


fn vec_func[
    op: fn (Float32, Float32) capturing [_] -> Float32
](
    in0: UnsafePointer[Float32],
    in1: UnsafePointer[Float32],
    output: UnsafePointer[Float32],
    len: Int,
):
    var tid = global_idx.x
    if tid >= UInt(len):
        return
    output[tid] = op(in0[tid], in1[tid])


# Force the capture to be captured instead of inlined away.
@no_inline
fn run_binary_add(ctx: DeviceContext, capture: Float32) raises:
    print("== run_binary_add")

    alias length = 1024

    var in0 = ctx.enqueue_create_buffer[DType.float32](length)
    var in1 = ctx.enqueue_create_buffer[DType.float32](length)
    var out = ctx.enqueue_create_buffer[DType.float32](length)

    with in0.map_to_host() as in0_host, in1.map_to_host() as in1_host:
        for i in range(length):
            in0_host[i] = i
            in1_host[i] = 2

    @parameter
    fn add(lhs: Float32, rhs: Float32) -> Float32:
        return capture + lhs + rhs

    var block_dim = 32
    alias kernel = vec_func[add]
    ctx.enqueue_function_checked[kernel, kernel](
        in0,
        in1,
        out,
        length,
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
    )
    print(
        "number of captures:",
        ctx.compile_function[vec_func[add]]()._func_impl.num_captures,
    )
    assert_equal(
        ctx.compile_function[vec_func[add]]()._func_impl.num_captures,
        1,
    )
    ctx.synchronize()

    with out.map_to_host() as out_host:
        expected: List[Float32] = [
            4.5,
            5.5,
            6.5,
            7.5,
            8.5,
            9.5,
            10.5,
            11.5,
            12.5,
            13.5,
        ]
        for i in range(10):
            print("at index", i, "the value is", out_host[i])
            assert_equal(out_host[i], expected[i])


def test_binary_apply():
    with DeviceContext() as ctx:
        run_binary_add(ctx, 2.5)


def main():
    # TODO(MOCO-2556): Use automatic discovery when it can handle global_idx.
    # TestSuite.discover_tests[__functions_in_module()]().run()
    var suite = TestSuite()

    suite.test[test_binary_apply]()

    suite^.run()
