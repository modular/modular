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
# Tests that gather with poison in an unmasked lane triggers abort.

from std.memory import UnsafePointer, alloc


# CHECK: use of uninitialized memory
def main():
    var ptr = alloc[Float32](4)

    # Initialize elements.
    ptr.store(0, Float32(1.0))
    ptr.store(1, Float32(2.0))
    ptr.store(2, Float32(3.0))
    ptr.store(3, Float32(4.0))

    # Poison element at index 2 with the debug allocator poison pattern
    # (FLT_MAX bits = 0x7F7FFFFF).
    (ptr + 2).bitcast[UInt32]().store(UInt32(0x7F7FFFFF))

    # Gather with offsets [0,1,2,3] — lane 2 is unmasked and poisoned.
    var offset = SIMD[DType.int64, 4](0, 1, 2, 3)
    var mask = SIMD[DType.bool, 4](True, True, True, False)
    _ = ptr.gather(offset=offset, mask=mask)

    # Should not reach here.
    ptr.free()
