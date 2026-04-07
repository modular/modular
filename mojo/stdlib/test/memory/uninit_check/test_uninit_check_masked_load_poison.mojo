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
# Tests that masked_load with poison in an unmasked lane triggers abort.

from std.memory import UnsafePointer, alloc
from std.sys.intrinsics import masked_load


# CHECK: use of uninitialized memory
def main():
    var ptr = alloc[Float32](4)

    # Initialize elements.
    ptr.store(0, Float32(1.0))
    ptr.store(1, Float32(2.0))
    ptr.store(2, Float32(3.0))
    ptr.store(3, Float32(4.0))

    # Poison element at index 1 (which will be unmasked = True = loaded).
    (ptr + 1).bitcast[UInt32]().store(UInt32(0xFFFFFFFF))

    # mask=True means "load from memory" — lane 1 is unmasked and poisoned.
    var mask = SIMD[DType.bool, 4](True, True, False, False)
    var passthrough = SIMD[DType.float32, 4](0)
    _ = masked_load(ptr, mask, passthrough)

    # Should not reach here.
    ptr.free()
