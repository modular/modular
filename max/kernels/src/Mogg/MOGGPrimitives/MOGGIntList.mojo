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
# A temporary home for the experimental int list type.


from buffer.dimlist import DimList, _make_tuple
from memory import memset_zero

from utils.index import IndexList


struct IntList[static_values: DimList = DimList()](
    Copyable, Defaultable, Movable, Sized
):
    # Array must be >= 1 length, so we clamp to that if we have unknown
    # length shape. DimList of size 0 represents a dynamically ranked list.
    alias _length = static_values.__len__()
    alias _safe_len = max(1, Self._length)

    # An alias to a parameter of the same sized shape as this but with the values unknown.
    alias _size_but_unknown = DimList() if Self._length == 0 else DimList.create_unknown[
        Self._safe_len
    ]()

    var data: UnsafePointer[Int]
    var stack_alloc_data: IndexList[Self._safe_len]
    var length: Int

    @always_inline
    fn __init__(out self):
        self.length = Self._length
        self.data = UnsafePointer[Int]()
        self.stack_alloc_data = IndexList[Self._safe_len]()

    # Should not be copy constructable, i.e passed by value, but can be cloned.
    @always_inline
    fn __init__(out self, other: IntList):
        var num_elements = len(other)
        self.length = Self._length
        self.data = UnsafePointer[Int]()
        self.stack_alloc_data = IndexList[Self._safe_len]()

        @parameter
        if Self.is_fully_static():
            # Fully static case stores nothing. Will just resolve to the
            # parameter values. Can just keep the default runtime values.
            return
        elif Self.has_static_length():
            # 2nd best case, we know the length but not all the values.
            # We can store the values in memory which doesn't need to be heap
            # allocated.
            @parameter
            for i in range(Self._length):
                self.stack_alloc_data[i] = other[i]
        else:
            # Worst case we allocate the memory on the heap.
            self.length = num_elements
            self.data = UnsafePointer[Int].alloc(num_elements)
            for i in range(num_elements):
                self.data[i] = other[i]
            self.stack_alloc_data = IndexList[Self._safe_len]()

    @always_inline
    fn __init__(out self, *elems: Int):
        var num_elements = len(elems)

        self.length = Self._length
        self.data = UnsafePointer[Int]()
        self.stack_alloc_data = IndexList[Self._safe_len]()

        @parameter
        if Self.is_fully_static():
            # Fully static case stores nothing. Will just resolve to the
            # parameter values. Can just keep the default runtime values.
            return
        elif Self.has_static_length():
            # 2nd best case, we know the length but not all the values.
            # We can store the values in memory which doesn't need to be heap
            # allocated.
            self.stack_alloc_data = IndexList[Self._safe_len]()

            @parameter
            for i in range(Self._length):
                self.stack_alloc_data[i] = elems[i]
        else:
            # Worst case we allocate the memory on the heap.
            self.length = num_elements
            self.data = UnsafePointer[Int].alloc(num_elements)
            for i in range(num_elements):
                self.data[i] = elems[i]
            self.stack_alloc_data = IndexList[Self._safe_len]()

    @always_inline
    fn __init__[rank: Int](out self, shape: IndexList[rank]):
        constrained[rank == len(static_values)]()
        self.length = rank
        self.data = UnsafePointer[Int]()
        self.stack_alloc_data = rebind[IndexList[Self._safe_len]](shape)

    @always_inline
    fn __copyinit__(out self, existing: Self):
        self.stack_alloc_data = existing.stack_alloc_data
        self.length = existing.length
        self.data = UnsafePointer[Int]()

        @parameter
        if not Self.has_static_length():
            self.data = UnsafePointer[Int].alloc(self.length)
            for i in range(self.length):
                self.data[i] = existing[i]

    @staticmethod
    @always_inline
    fn zeros(length: Int) -> Self:
        """
        Create a new IntList with all zeros.
        """
        var new = Self()

        @parameter
        if Self.has_static_length():

            @parameter
            for i in range(Self._length):
                new.stack_alloc_data[i] = 0
        else:
            # Worst case we allocate the memory on the heap.
            new.length = length
            new.data = UnsafePointer[Int].alloc(length)
            memset_zero(new.data, length)
        return new

    @staticmethod
    @always_inline
    fn shape_idx_statically_known[idx: Int]() -> Bool:
        @parameter
        if not Self.has_static_length():
            return False
        else:
            return Self.static_values.at[idx]().has_value()

    @staticmethod
    @always_inline
    fn is_fully_static() -> Bool:
        @parameter
        if not Self.has_static_length():
            return False
        else:
            return Self.static_values.all_known[Self._length]()

    @staticmethod
    @always_inline
    fn has_static_length() -> Bool:
        return Self._length != 0

    @always_inline
    fn to_static_tuple(self) -> IndexList[Self._safe_len]:
        constrained[
            Self.has_static_length(),
            (
                "IntList must have statically known length to be converted into"
                " static tuple"
            ),
        ]()

        @parameter
        if Self.is_fully_static():
            return _make_tuple[Self._safe_len](Self.static_values)
        else:
            return self.stack_alloc_data

    @always_inline
    fn __getitem__(self, index: Int) -> Int:
        """Gets the value at the specified index.

        Args:
          index: The index of the value to retrieve.

        Returns:
          The value at the specified indices.
        """

        @parameter
        if Self.is_fully_static():
            var v = _make_tuple[Self._length](Self.static_values)
            return v[index]
        elif Self.has_static_length():
            return self.stack_alloc_data[index]
        else:
            return self.data[index]

    @always_inline
    fn nelems(self) -> Int:
        var num_elms: Int = 1

        @parameter
        if Self.has_static_length():

            @parameter
            for i in range(Self._length):
                num_elms *= self[i]

        else:
            for i in range(len(self)):
                num_elms *= self[i]
        return num_elms

    @always_inline("nodebug")
    fn __setitem__(mut self, index: Int, value: Int):
        constrained[
            not Self.is_fully_static(),
            "Fully static int lists can't be modified",
        ]()

        @parameter
        if Self.has_static_length():
            self.stack_alloc_data[index] = value
        else:
            self.data[index] = value

    @staticmethod
    @always_inline
    fn empty(length: Int) -> IntList[Self._size_but_unknown]:
        var x = IntList[Self._size_but_unknown]()
        x.length = length

        @parameter
        if not Self.has_static_length():
            x.data = UnsafePointer[Int].alloc(length)
            for i in range(length):
                x.data[i] = 0
        return x^

    fn print(self):
        var result: String = "("
        for i in range(len(self)):
            result += String(self[i])
            if i != (len(self) - 1):
                result += ", "
        result += ")"
        print(result)

    @always_inline
    fn __len__(self) -> Int:
        @parameter
        if Self.has_static_length():
            return Self._length
        return self.length

    @always_inline
    fn __del__(deinit self):
        @parameter
        if not Self.has_static_length():
            if self.data != UnsafePointer[Int]():
                self.data.free()
