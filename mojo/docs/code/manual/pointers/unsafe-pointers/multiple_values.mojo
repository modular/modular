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


def main():
    # start-alloc-multiple
    var float_ptr = alloc[Float64](6)
    for offset in range(6):
        (float_ptr + offset).init_pointee_copy(0.0)
    # end-alloc-multiple

    # start-subscript-access
    float_ptr[2] = 3.0
    for offset in range(6):
        print(float_ptr[offset], end=", ")
    # end-subscript-access
    print()

    # Pointer arithmetic: offset from an existing pointer
    var first_ptr = float_ptr
    # start-pointer-offset
    var third_ptr = first_ptr + 2
    # end-pointer-offset

    print(third_ptr[])

    # In-place pointer advance
    var ptr = float_ptr
    # start-pointer-advance
    # Advance the pointer one element:
    ptr += 1
    # end-pointer-advance

    print(ptr[])

    for offset in range(6):
        (float_ptr + offset).destroy_pointee()
    float_ptr.free()
