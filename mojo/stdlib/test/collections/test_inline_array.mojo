# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from sys.info import size_of

from memory.maybe_uninitialized import UnsafeMaybeUninitialized
from test_utils import CopyCounter, DelRecorder, MoveCounter
from testing import assert_equal, assert_true


def test_array_unsafe_get():
    # Negative indexing is undefined behavior with unsafe_get
    # so there are not test cases for it.
    var arr = InlineArray[Int, 3](0, 0, 0)

    assert_equal(arr.unsafe_get(0), 0)
    assert_equal(arr.unsafe_get(1), 0)
    assert_equal(arr.unsafe_get(2), 0)

    arr[0] = 1
    arr[1] = 2
    arr[2] = 3

    assert_equal(arr.unsafe_get(0), 1)
    assert_equal(arr.unsafe_get(1), 2)
    assert_equal(arr.unsafe_get(2), 3)


def test_array_int():
    var arr = InlineArray[Int, 3](0, 0, 0)

    assert_equal(arr[0], 0)
    assert_equal(arr[1], 0)
    assert_equal(arr[2], 0)

    arr[0] = 1
    arr[1] = 2
    arr[2] = 3

    assert_equal(arr[0], 1)
    assert_equal(arr[1], 2)
    assert_equal(arr[2], 3)

    # test negative indexing
    assert_equal(arr[-1], 3)
    assert_equal(arr[-2], 2)

    # test negative indexing with dynamic index
    var i = -1
    assert_equal(arr[i], 3)
    i -= 1
    assert_equal(arr[i], 2)

    var copy = arr
    assert_equal(arr[0], copy[0])
    assert_equal(arr[1], copy[1])
    assert_equal(arr[2], copy[2])

    var move = arr^
    assert_equal(copy[0], move[0])
    assert_equal(copy[1], move[1])
    assert_equal(copy[2], move[2])

    # fill element initializer
    var arr2 = InlineArray[Int, 3](fill=5)
    assert_equal(arr2[0], 5)
    assert_equal(arr2[1], 5)
    assert_equal(arr2[2], 5)

    var arr3 = InlineArray[Int, 1](5)
    assert_equal(arr3[0], 5)

    def test_init_fill[size: Int, batch_size: Int, dt: DType](arg: Scalar[dt]):
        var arr = InlineArray[Scalar[dt], size].__init__[batch_size=batch_size](
            fill=arg
        )
        for i in range(size):
            assert_equal(arr[i], arg)

    def test_init_fill_scalars[
        *dts: DType, sizes: List[Int], batch_sizes: List[Int]
    ]():
        @parameter
        for current_batch_size in range(len(batch_sizes)):

            @parameter
            for current_size in range(len(sizes)):

                @parameter
                for current_type in range(len(VariadicList(dts))):
                    test_init_fill[
                        sizes[current_size], batch_sizes[current_batch_size]
                    ](Scalar[dts[current_type]].MAX)

    test_init_fill_scalars[
        Int64.dtype,
        Int8.dtype,
        sizes= [1, 32, 64, 129, 256, 512, 768, 1000],
        batch_sizes= [1, 8, 32, 64, 128],
    ]()

    test_init_fill[2048, 512](Int64.MAX)
    test_init_fill[2048, 1](Int64.MAX)


def test_array_str():
    var arr: InlineArray[String, 3] = ["hi", "hello", "hey"]

    assert_equal(arr[0], "hi")
    assert_equal(arr[1], "hello")
    assert_equal(arr[2], "hey")

    # Test mutating an array through its __getitem__
    arr[0] = "howdy"
    arr[1] = "morning"
    arr[2] = "wazzup"

    assert_equal(arr[0], "howdy")
    assert_equal(arr[1], "morning")
    assert_equal(arr[2], "wazzup")

    # test negative indexing
    assert_equal(arr[-1], "wazzup")
    assert_equal(arr[-2], "morning")

    var copy = arr
    assert_equal(arr[0], copy[0])
    assert_equal(arr[1], copy[1])
    assert_equal(arr[2], copy[2])

    var move = arr^
    assert_equal(copy[0], move[0])
    assert_equal(copy[1], move[1])
    assert_equal(copy[2], move[2])

    # fill element initializer
    var arr2 = InlineArray[String, 3](fill="hi")
    assert_equal(arr2[0], "hi")
    assert_equal(arr2[1], "hi")
    assert_equal(arr2[2], "hi")

    # size 1 array to prevent regressions in the constructors
    var arr3 = InlineArray[String, 1]("hi")
    assert_equal(arr3[0], "hi")


def test_array_int_pointer():
    var arr = InlineArray[Int, 3](0, 10, 20)

    var ptr = arr.unsafe_ptr()
    assert_equal(ptr[0], 0)
    assert_equal(ptr[1], 10)
    assert_equal(ptr[2], 20)

    ptr[0] = 0
    ptr[1] = 1
    ptr[2] = 2

    assert_equal(arr[0], 0)
    assert_equal(arr[1], 1)
    assert_equal(arr[2], 2)

    assert_equal(ptr[0], 0)
    assert_equal(ptr[1], 1)
    assert_equal(ptr[2], 2)

    # We make sure it lives long enough
    _ = arr


def test_array_unsafe_assume_initialized_constructor_string():
    var maybe_uninitialized_arr = InlineArray[
        UnsafeMaybeUninitialized[String], 3
    ](uninitialized=True)
    maybe_uninitialized_arr[0].write("hello")
    maybe_uninitialized_arr[1].write("mojo")
    maybe_uninitialized_arr[2].write("world")

    var initialized_arr = InlineArray[String, 3](
        unsafe_assume_initialized=maybe_uninitialized_arr^
    )

    assert_equal(initialized_arr[0], "hello")
    assert_equal(initialized_arr[1], "mojo")
    assert_equal(initialized_arr[2], "world")

    # trigger a move
    var initialized_arr2 = initialized_arr^

    assert_equal(initialized_arr2[0], "hello")
    assert_equal(initialized_arr2[1], "mojo")
    assert_equal(initialized_arr2[2], "world")

    # trigger a copy
    var initialized_arr3 = initialized_arr2.copy()

    assert_equal(initialized_arr3[0], "hello")
    assert_equal(initialized_arr3[1], "mojo")
    assert_equal(initialized_arr3[2], "world")

    # We assume the destructor was called correctly, but one
    # might want to add a test for that in the future.


def test_array_contains():
    var arr = InlineArray[String, 3]("hi", "hello", "hey")
    assert_true("hi" in arr)
    assert_true(not "greetings" in arr)


def test_inline_array_runs_destructors():
    """Ensure we delete the right number of elements."""
    var destructor_counter = List[Int]()
    var pointer_to_destructor_counter = UnsafePointer(to=destructor_counter)
    alias capacity = 32
    var inline_list = InlineArray[DelRecorder, 4](
        DelRecorder(0, pointer_to_destructor_counter),
        DelRecorder(10, pointer_to_destructor_counter),
        DelRecorder(20, pointer_to_destructor_counter),
        DelRecorder(30, pointer_to_destructor_counter),
    )
    _ = inline_list
    # This is the last use of the inline list, so it should be destroyed here,
    # along with each element.
    assert_equal(len(destructor_counter), 4)
    assert_equal(destructor_counter[0], 0)
    assert_equal(destructor_counter[1], 10)
    assert_equal(destructor_counter[2], 20)
    assert_equal(destructor_counter[3], 30)


fn test_unsafe_ptr() raises:
    alias N = 10
    var arr = InlineArray[Int, 10](fill=0)
    for i in range(N):
        arr[i] = i

    var ptr = arr.unsafe_ptr()
    for i in range(N):
        assert_equal(arr[i], ptr[i])


def test_size_of_array[current_type: Copyable & Movable, capacity: Int]():
    """Testing if `size_of` the array equals capacity * `size_of` current_type.

    Parameters:
        current_type: The type of the elements of the `InlineList`.
        capacity: The capacity of the `InlineList`.
    """
    alias size_of_current_type = size_of[current_type]()
    assert_equal(
        size_of[InlineArray[current_type, capacity]](),
        capacity * size_of_current_type,
    )


def test_move():
    """Test that moving an InlineArray works correctly."""

    # === 1. Check that the move constructor is called correctly. ===

    var arr = InlineArray[MoveCounter[Int], 3]({1}, {2}, {3})
    var copied_arr = arr.copy()

    for i in range(len(arr)):
        # The elements were moved into the array
        assert_equal(arr[i].move_count, 1)

    var moved_arr = arr^

    for i in range(len(moved_arr)):
        # Check that the moved array has the same elements as the copied array
        assert_equal(copied_arr[i].value, moved_arr[i].value)
        # Check that the move constructor was called again for each element
        assert_equal(moved_arr[i].move_count, 2)

    # === 2. Check that the copy constructor is not called when moving. ===

    var arr2 = InlineArray[CopyCounter, 3]({}, {}, {})
    for i in range(len(arr2)):
        # The elements were moved into the array and not copied
        assert_equal(arr2[i].copy_count, 0)

    var moved_arr2 = arr2^

    for i in range(len(moved_arr2)):
        # Check that the copy constructor was not called
        assert_equal(moved_arr2[i].copy_count, 0)

    # === 3. Check that the destructor is not called when moving. ===

    var destructor_counter = List[Int]()
    var pointer_to_destructor_counter = UnsafePointer(to=destructor_counter)
    var del_recorder = DelRecorder(0, pointer_to_destructor_counter)
    var arr3 = InlineArray[DelRecorder, 1](del_recorder)

    assert_equal(len(pointer_to_destructor_counter[]), 0)

    var moved_arr3 = arr3^

    assert_equal(len(pointer_to_destructor_counter[]), 0)

    _ = moved_arr3

    # Double check that the destructor is called when the array is destroyed
    assert_equal(len(pointer_to_destructor_counter[]), 1)
    _ = del_recorder


def main():
    test_array_unsafe_get()
    test_array_int()
    test_array_str()
    test_array_int_pointer()
    test_array_unsafe_assume_initialized_constructor_string()
    test_array_contains()
    test_inline_array_runs_destructors()
    test_unsafe_ptr()
    test_size_of_array[Int, capacity=32]()
    test_move()
