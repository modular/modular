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
# Tests that a 4-element SIMD vector load where one element is poisoned
# triggers abort.

from std.memory import UnsafePointer, alloc


# CHECK: use of uninitialized memory
def main():
    var ptr = alloc[Float32](4)

    # Initialize all elements to safe values.
    ptr.store(0, Float32(1.0))
    ptr.store(1, Float32(2.0))
    ptr.store(2, Float32(3.0))
    ptr.store(3, Float32(4.0))

    # Poison just one element (index 2) with host poison (0xFF fill).
    (ptr + 2).bitcast[UInt32]().store(UInt32(0xFFFFFFFF))

    # Loading a 4-wide SIMD vector should detect the poisoned element.
    _ = ptr.load[width=4]()

    # Should not reach here.
    ptr.free()
