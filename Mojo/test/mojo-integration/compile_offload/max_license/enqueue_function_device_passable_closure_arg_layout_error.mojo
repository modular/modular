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

# `%mojo-build` under asan adds `--sanitize address`; GPU DeviceContext
# codegen then runs for minutes and FileCheck never sees the comptime note.
# UNSUPPORTED: asan

# RUN: not %mojo-build --target-accelerator="nvidia:sm_90a" %s -o %t 2>&1 | FileCheck %s --dump-input=fail

# The kernel reads a closure argument in its host layout, but enqueue writes the
# closure's `device_type`. When a capture's device layout is smaller than its
# host layout (here `HostString` drops `host_padding`), the kernel would read
# past the encoded bytes, so the launch check must reject it.
#
# CHECK: constraint failed: argument #0 of type '{{.*}}read_fn{{.*}}' encodes to 16 bytes but the declared function argument type '{{.*}}read_fn{{.*}}' is 32 bytes.
# CHECK-SAME: If the argument's host bytes are already valid on the device, pass it via `host_args=` to skip encoding

from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from max.gpu.host import DeviceContext


@fieldwise_init
struct HostString(DevicePassable, ImplicitlyCopyable, TrivialRegisterPassable):
    comptime device_type: AnyType = SIMD[.int32, 4]
    var value: SIMD[.int32, 4]
    var host_padding: SIMD[.int32, 4]

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self.value, target)

    @staticmethod
    def get_type_name() -> String:
        return "HostString"


def apply_closure[
    closure_type: (def() -> None) & DevicePassable & ImplicitlyCopyable
](closure: closure_type):
    closure()


def main() raises:
    var ctx = DeviceContext()
    var capture = HostString(SIMD[.int32, 4](1), SIMD[.int32, 4](0))

    def read_fn() {var capture}:
        _ = capture

    comptime kernel = apply_closure[type_of(read_fn)]
    var compiled_kernel = ctx.compile_function[kernel]()
    ctx.enqueue_function(compiled_kernel, read_fn, grid_dim=1, block_dim=1)
