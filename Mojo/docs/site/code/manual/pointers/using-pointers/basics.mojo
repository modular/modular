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
from std.memory.alloc import alloc, dealloc, Layout
from std.testing import assert_equal


def test_intro():
    # start-safe-example
    var count: Int = 0
    # Point to an existing value
    var ptr = Pointer(to=count)  # ptr's type is Pointer[Int, ...]
    # Mutate the value
    ptr[] = 100
    # end-safe-example


def test_basics():
    # Allocate memory to hold a value
    # start-basics-alloc
    var allocation = alloc(Layout[Int].single())
    var ptr = allocation.unsafe_ptr()
    # end-basics-alloc
    # Initialize the allocated memory
    ptr.unsafe_write(100)

    # start-dereference
    # Update an initialized value
    ptr[] += 10
    # Access an initialized value
    print(ptr[])
    # end-dereference

    dealloc(allocation^)


def test_pointer_to_value():
    # start-pointer-to-value
    var counter: Int = 5
    var ptr = Pointer(to=counter)
    # end-pointer-to-value

    # start-dereference-read-mutate
    # Read from pointee
    print(ptr[])
    # Mutate pointee
    ptr[] = 0
    # end-dereference-read-mutate


def test_alloc_string():
    # start-alloc-string
    var allocation = alloc[String]({count = 1})
    var str_ptr = allocation.unsafe_ptr()
    # str_ptr[] = "Testing" # Undefined behavior!
    str_ptr.unsafe_write("Testing")
    str_ptr[] += " pointers"  # Works now
    # end-alloc-string

    str_ptr.unsafe_deinit_pointee()
    dealloc(allocation^)


def test_unsafe_write_owned():
    var allocation = alloc[String]({count = 1})
    var str_ptr = allocation.unsafe_ptr()
    # start-unsafe-write-owned
    str_ptr.unsafe_write("Owned string")
    # end-unsafe-write-owned

    str_ptr.unsafe_deinit_pointee()
    dealloc(allocation^)


def test_unsafe_write_newly_copied() raises:
    var source = String("Hello")
    var allocation = alloc[String]({count = 1})
    var str_ptr = allocation.unsafe_ptr()
    # A newly copied value is unowned too, so it needs no transfer sigil.
    str_ptr.unsafe_write(source.copy())
    var written = str_ptr[].copy()
    str_ptr.unsafe_deinit_pointee()
    dealloc(allocation^)

    # Assert after deallocating: a raising call between `alloc` and
    # `dealloc` would abandon the allocation on the throw path.
    assert_equal(written, "Hello")
    assert_equal(source, "Hello")  # the copy left the source intact


def test_pointer_to_string():
    # start-pointer-to-string
    var s = "Testing"
    var s_ptr = Pointer(to=s)
    # end-pointer-to-string

    _ = s_ptr


def main() raises:
    test_intro()
    test_basics()
    test_pointer_to_value()
    test_alloc_string()
    test_unsafe_write_owned()
    test_unsafe_write_newly_copied()
    test_pointer_to_string()
