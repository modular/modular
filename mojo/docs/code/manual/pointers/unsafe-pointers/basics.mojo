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


def test_alloc_and_init():
    # start-alloc-and-init
    # Allocate memory to hold a value
    var ptr = alloc[Int](1)
    # Initialize the allocated memory
    ptr.init_pointee_copy(100)
    # end-alloc-and-init

    # start-dereference
    # Update an initialized value
    ptr[] += 10
    # Access an initialized value
    print(ptr[])
    # end-dereference

    ptr.free()


def test_pointer_to_value():
    # start-pointer-to-value
    var counter: Int = 5
    var ptr = UnsafePointer(to=counter)
    # end-pointer-to-value

    # start-dereference-read-mutate
    # Read from pointee
    print(ptr[])
    # mutate pointee
    ptr[] = 0
    # end-dereference-read-mutate


def test_alloc_string():
    # start-alloc-string
    var str_ptr = alloc[String](1)
    # str_ptr[] = "Testing" # Undefined behavior!
    str_ptr.init_pointee_move("Testing")
    str_ptr[] += " pointers"  # Works now
    # end-alloc-string

    str_ptr.destroy_pointee()
    str_ptr.free()


def test_init_pointee_move_owned():
    var str_ptr = alloc[String](1)
    # start-init-pointee-move-owned
    str_ptr.init_pointee_move("Owned string")
    # end-init-pointee-move-owned

    str_ptr.destroy_pointee()
    str_ptr.free()


def test_pointer_to_string():
    # start-pointer-to-string
    s = "Testing"
    s_ptr = UnsafePointer(to=s)
    # end-pointer-to-string

    _ = s_ptr


def main():
    test_alloc_and_init()
    test_pointer_to_value()
    test_alloc_string()
    test_init_pointee_move_owned()
    test_pointer_to_string()
