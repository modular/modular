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

from std.gpu.host import DeviceContext


# CHECK-LABEL: test_metal_print_basic
def test_metal_print_basic() raises:
    print("test_metal_print_basic")

    def do_print():
        print("Hello from Metal GPU")

    with DeviceContext() as ctx:
        ctx.enqueue_function_experimental[do_print](grid_dim=1, block_dim=1)
        ctx.synchronize()

    # CHECK: Hello from Metal GPU


# CHECK-LABEL: test_metal_print_int
def test_metal_print_int() raises:
    print("test_metal_print_int")

    def do_print(x: Int):
        print("x =", x)

    with DeviceContext() as ctx:
        ctx.enqueue_function_experimental[do_print](
            Int(42), grid_dim=1, block_dim=1
        )
        ctx.synchronize()

    # CHECK: x = 42


# CHECK-LABEL: test_metal_print_float32
def test_metal_print_float32() raises:
    print("test_metal_print_float32")

    # Note: Apple GPU does not support Float64. Use Float32.
    def do_print(y: Float32):
        print("y =", y)

    with DeviceContext() as ctx:
        ctx.enqueue_function_experimental[do_print](
            Float32(3.14), grid_dim=1, block_dim=1
        )
        ctx.synchronize()

    # CHECK: y = 3.14{{[0-9]*}}


# CHECK-LABEL: test_metal_print_empty
def test_metal_print_empty() raises:
    print("test_metal_print_empty")

    def do_print():
        print("")

    with DeviceContext() as ctx:
        ctx.enqueue_function_experimental[do_print](grid_dim=1, block_dim=1)
        ctx.synchronize()


def main() raises:
    test_metal_print_basic()
    test_metal_print_int()
    test_metal_print_float32()
    test_metal_print_empty()
