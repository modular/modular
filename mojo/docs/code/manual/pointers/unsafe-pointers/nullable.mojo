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


def test_unsafe_dangling():
    # start-unsafe-dangling
    var ptr = UnsafePointer[Int, MutUntrackedOrigin].unsafe_dangling()
    # end-unsafe-dangling

    _ = ptr


def main():
    # start-optional-pointer
    var ptr = Optional[UnsafePointer[Int, MutUntrackedOrigin]]()
    # end-optional-pointer

    # Optional is initially None; if block is skipped.
    if ptr:
        # ptr is not None — safe to unwrap
        var p = ptr.value()
        print(p[])

    # Assign a real value to test the non-None path.
    var x_ptr = alloc[Int](1)
    x_ptr.init_pointee_copy(42)
    ptr = x_ptr
    if ptr:
        # ptr is not None — safe to unwrap
        var p = ptr.value()
        print(p[])
    x_ptr.destroy_pointee()
    x_ptr.free()

    test_unsafe_dangling()
