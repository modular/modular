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

# RUN: %mojo-build --target-accelerator="nvidia:sm_90a" %s -o %t

from std.builtin.device_passable import DevicePassable
from max.gpu.host import DeviceContext


def call_closure[
    closure_type: (def() -> None) & DevicePassable & ImplicitlyCopyable
](closure: closure_type):
    closure()


def main() raises:
    var ctx = DeviceContext()
    var captured = Int32(42)

    def closure() {var captured}:
        _ = captured

    comptime kernel = call_closure[type_of(closure)]
    var compiled_kernel = ctx.compile_function[kernel]()
    ctx.enqueue_function(
        compiled_kernel,
        closure,
        grid_dim=1,
        block_dim=1,
    )
