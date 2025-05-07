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
"""
Implements the `TensorShape` type - a space efficient representation of a
tensor shape.

You can import these APIs from the `tensor` package. For example:

```mojo
from max.tensor import TensorShape
```
"""


from collections import List
from sys.info import is_little_endian, sizeof
from sys.intrinsics import _type_is_eq

from memory import UnsafePointer, memcmp, memcpy

from utils.index import IndexList
from utils.static_tuple import StaticTuple

# These representation must be kept in sync with the TensorShape file in
# Support/include/Support/ML/TensorShape.h

# These representation must be kept in sync with the TensorShape file in
# Support/include/Support/ML/TensorShape.h

# This supports two inline representations and an out-of-line one:
#  1) k16 can hold up to 6 dimensions when they fit into 16-bits.
#  2) k32 can hold up to 4 dimension where the first three fits in
#     32-bits and the last fits in 8 bits (typically channels or batch
#     size).
#  3) kOutOfLine is used for the general case.
#
# Important: Identical shapes have the same representation kind to allow
# efficient shape comparison with memcmp for k16 and k32.
#
# Each representation has an additional 8 bits of unused "auxillary"
# storage.  This is used to hold a DType for TensorSpec.  We keep
# this at the end of the storage so we can efficiently omit it from
# memset/memcpy operations.


# ===-----------------------------------------------------------------------===#
# Utilities
# ===-----------------------------------------------------------------------===#

# TODO(#20271): Support dynamically ranked shapes as per the C++ runtime.


@register_passable("trivial")
struct _RepKind(EqualityComparable):
    alias KIND_16 = _RepKind(0)
    """A representation which can hold up to 6 dimensions with each dim
    occupying at most 16-bits."""
    alias KIND_32 = _RepKind(1)
    """A representation which can hold up to 4 dimensions with the first three
    occupying at most 32-bits and the last occupying at most 8 bits."""
    alias KIND_OUT_OF_LINE = _RepKind(2)
    """A general storage kind which stores the dimensions on the heap."""

    var kind: UInt8

    @always_inline("nodebug")
    @implicit
    fn __init__(out self, value: Int):
        self.kind = value

    @always_inline("nodebug")
    fn __eq__(self, rhs: _RepKind) -> Bool:
        return (self.kind == rhs.kind).__bool__()

    @always_inline("nodebug")
    fn __ne__(self, rhs: _RepKind) -> Bool:
        return not self == rhs


@register_passable("trivial")
struct _Rep16(Stringable, Writable, EqualityComparable):
    """A representation which can hold up to 6 dimensions with each dim
    occupying at most 16-bits."""

    var dims: StaticTuple[Int16, 6]
    """The dimensions."""
    var _unused: UInt8
    """Unused block."""
    var rep_kind: _RepKind
    """The representation kind."""
    var rank: UInt8
    """The rank of the shape."""
    var auxillary: UInt8
    """Auxillary information about the shape."""

    @always_inline
    fn __init__(out self):
        """Default initializes the _Rep16 type."""
        self.dims = StaticTuple[Int16, 6]()
        self._unused = 0
        self.rep_kind = _RepKind.KIND_16
        self.rank = 0
        self.auxillary = 0

    @always_inline
    fn __eq__(self, other: Self) -> Bool:
        """Returns True if the two values are the same and False otherwise.

        Args:
          other: The other _Rep16 to compare against.

        Returns:
          True if the two shapes are the same and False otherwise.
        """
        if self.get_rank() != other.get_rank():
            return False

        for i in range(self.get_rank()):
            if self[i] != other[i]:
                return False

        return True

    @always_inline
    fn __ne__(self, other: Self) -> Bool:
        """Returns True if the two values are not the same and False otherwise.

        Args:
          other: The other _Rep16 to compare against.

        Returns:
          True if the two shapes are the not the same and False otherwise.
        """

        return not (self == other)

    @always_inline
    fn get_rank(self) -> Int:
        """Gets the rank of the representation.

        Returns:
          The rank of the representation.
        """
        debug_assert(Int(self.rank) <= 6, "index out of range")
        return Int(self.rank)

    @always_inline
    fn get_num_elements(self) -> Int:
        """Gets the number of elements of the representation.

        Returns:
          The number of elements in the representation.
        """
        var rank = self.get_rank()
        var product: Int = 1
        for i in range(rank):
            product *= Int(self.dims[i])
        return product

    @always_inline
    fn __getitem__(self, index: Int) -> Int:
        """Gets dimension at the specified index.

        Args:
          index: the dimension index.

        Returns:
          The value at the specified dimension.
        """
        return Int(self.dims[index])

    @always_inline
    fn __setitem__(mut self, index: Int, val: Int):
        """Sets the dimension at the specified index.

        Args:
          index: the dimension index.
          val: the value to set.
        """
        debug_assert(index < self.get_rank(), "index out of range")
        self.dims[index] = val

    @always_inline
    fn __repr__(self) -> String:
        """Returns the string representation of the shape.

        Returns:
          The string representation of the shape.
        """
        return self.__str__()

    @always_inline
    fn __str__(self) -> String:
        """Returns the string representation of the shape.

        Returns:
          The string representation of the shape.
        """
        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
        for i in range(self.get_rank()):
            if i != 0:
                writer.write("x")
            writer.write(self[i])


@register_passable("trivial")
struct _Rep32(Writable, EqualityComparable):
    """A representation which can hold up to 4 dimensions with the first three
    occupying at most 32-bits and the last occupying at most 8 bits."""

    var dims012: StaticTuple[Int32, 3]  # dim0, dim1, dim2
    """The 3 leading dimensions."""
    var dim3: Int8
    """The trailing dimension."""
    var rep_kind: _RepKind
    """The representation kind."""
    var rank: UInt8
    """The rank of the shape."""
    var auxillary: UInt8
    """Auxillary information about the shape."""

    @always_inline
    fn __init__(out self):
        """Default initializes the _Rep32 type."""
        self.dims012 = StaticTuple[Int32, 3]()
        self.dim3 = 0
        self.rep_kind = _RepKind.KIND_32
        self.rank = 0
        self.auxillary = 0

    @always_inline
    fn __eq__(self, other: Self) -> Bool:
        """Returns True if the two values are the same and False otherwise.

        Args:
          other: The other _Rep32 to compare against.

        Returns:
          True if the two shapes are the same and False otherwise.
        """
        if self.get_rank() != other.get_rank():
            return False

        for i in range(min(self.get_rank(), 3)):
            if self.dims012[i] != other.dims012[i]:
                return False

        if self.get_rank() > 3 and self.dim3 != other.dim3:
            return False

        return True

    @always_inline
    fn __ne__(self, other: Self) -> Bool:
        """Returns True if the two values are not the same and False otherwise.

        Args:
          other: The other _Rep32 to compare against.

        Returns:
          True if the two shapes are the not the same and False otherwise.
        """

        return not (self == other)

    @always_inline
    fn get_rank(self) -> Int:
        """Gets the rank of the representation.

        Returns:
          The rank of the representation.
        """
        debug_assert(Int(self.rank) <= 4, "index out of range")
        return Int(self.rank)

    @always_inline
    fn get_num_elements(self) -> Int:
        """Gets the number of elements of the representation.

        Returns:
          The number of elements in the representation.
        """
        var rank = self.get_rank()
        var product: Int = 1
        if rank == 4:
            product = Int(self.dim3)
            rank -= 1
        for i in range(rank):
            product *= Int(self.dims012[i])
        return product

    @always_inline
    fn __getitem__(self, index: Int) -> Int:
        """Gets dimension at the specified index.

        Args:
          index: the dimension index.

        Returns:
          The value at the specified dimension.
        """
        debug_assert(index <= 3, "index out of range")
        if index == 3:
            return Int(self.dim3)
        else:
            return Int(self.dims012[index])

    @always_inline
    fn __setitem__(mut self, index: Int, val: Int):
        """Sets the dimension at the specified index.

        Args:
          index: the dimension index.
          val: the value to set.
        """
        debug_assert(index < self.get_rank(), "index out of range")
        if index == 3:
            self.dim3 = val
        else:
            self.dims012[index] = val

    @always_inline
    fn __repr__(self) -> String:
        """Returns the string representation of the shape.

        Returns:
          The string representation of the shape.
        """
        return self.__str__()

    @always_inline
    fn __str__(self) -> String:
        """Returns the string representation of the shape.

        Returns:
          The string representation of the shape.
        """

        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
        for i in range(self.get_rank()):
            if i != 0:
                writer.write("x")
            writer.write(self[i])


@register_passable("trivial")
struct _RepOutOfLine(Writable, EqualityComparable):
    """A general storage kind which stores the dimensions on the heap."""

    alias _padding_size = (13 - sizeof[UnsafePointer[NoneType]]())
    var dims: UnsafePointer[Scalar[DType.index]]
    """The heap allocated dimensions."""
    # FIXME: This isn't correct for big endian systems, but we check with
    # static_assert below.
    var _padding: StaticTuple[UInt8, Self._padding_size]
    """Unused padding value."""
    var rep_kind: _RepKind
    """The representation kind."""
    var rank: UInt8
    """The rank of the shape."""
    var auxillary: UInt8
    """Auxillary information about the shape."""

    @always_inline
    fn __init__(out self):
        """Default initializes the _RepOutOfLine type."""
        constrained[
            is_little_endian(),
            (
                "the out of line representation is only implemented on little"
                " endian systems"
            ),
        ]()
        self.dims = UnsafePointer[Scalar[DType.index]]()
        self._padding = StaticTuple[UInt8, Self._padding_size]()
        self.rep_kind = _RepKind.KIND_OUT_OF_LINE
        self.rank = 0
        self.auxillary = 0

    @always_inline
    fn __eq__(self, other: Self) -> Bool:
        """Returns True if the two values are the same and False otherwise.

        Args:
          other: The other _RepOutOfLine to compare against.

        Returns:
          True if the two shapes are the same and False otherwise.
        """
        if self.get_rank() != other.get_rank():
            return False

        return memcmp(self.dims, other.dims, self.get_rank()) == 0

    @always_inline
    fn __ne__(self, other: Self) -> Bool:
        """Returns True if the two values are not the same and False otherwise.

        Args:
          other: The other _RepOutOfLine to compare against.

        Returns:
          True if the two shapes are the not the same and False otherwise.
        """

        return not (self == other)

    @always_inline
    fn get_rank(self) -> Int:
        """Gets the rank of the representation.

        Returns:
          The rank of the representation.
        """
        return Int(self.rank)

    @always_inline
    fn __getitem__(self, index: Int) -> Int:
        """Gets dimension at the specified index.

        Args:
          index: the dimension index.

        Returns:
          The value at the specified dimension.
        """
        return Int(self.dims.load(index))

    @always_inline
    fn __setitem__(mut self, index: Int, val: Int):
        """Sets the dimension at the specified index.

        Args:
          index: the dimension index.
          val: the value to set.
        """
        debug_assert(index < self.get_rank(), "index out of range")
        self.dims[index] = val

    @always_inline
    fn get_num_elements(self) -> Int:
        """Gets the number of elements of the representation.

        Returns:
          The number of elements in the representation.
        """
        var prod: Int = 1
        for i in range(self.get_rank()):
            prod *= self[i]
        return prod

    @always_inline
    fn copy(self) -> Self:
        """Creates a new copy of the object. Note that this will cause a heap
        allocation.

        Returns:
          A new copy of the representation.
        """
        var result = Self()
        result.dims = UnsafePointer[Scalar[DType.index]].alloc(self.get_rank())
        memcpy(result.dims, self.dims, self.get_rank())

        result._padding = self._padding
        result.rep_kind = self.rep_kind
        result.rank = self.rank
        result.auxillary = self.auxillary
        return result

    @always_inline
    fn __repr__(self) -> String:
        """Returns the string representation of the shape.

        Returns:
          The string representation of the shape.
        """
        return self.__str__()

    @always_inline
    fn __str__(self) -> String:
        """Returns the string representation of the shape.

        Returns:
          The string representation of the shape.
        """

        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
        for i in range(self.get_rank()):
            if i != 0:
                writer.write("x")
            writer.write(self[i])


@register_passable("trivial")
struct _TensorShapeStorage:
    """The storage type for the tensor shape. This acts as a union type between
    all the representations."""

    var ptr: UnsafePointer[NoneType]
    var idx: Int64

    @always_inline
    fn __init__(out self):
        """Default initializes the _TensorShapeStorage type."""
        var rep = _Rep32()
        var rep_ptr = UnsafePointer(to=rep)
        self = rep_ptr.bitcast[_TensorShapeStorage]()[]

    @always_inline
    @implicit
    fn __init__(out self, rep: _Rep16):
        """Initializes the _TensorShapeStorage from a _Rep16.

        Args:
          rep: A shape representation.
        """
        var rep_copy = rep
        var rep_ptr = UnsafePointer(to=rep_copy)
        self = rep_ptr.bitcast[_TensorShapeStorage]()[]

    @always_inline
    @implicit
    fn __init__(out self, rep: _Rep32):
        """Initializes the _TensorShapeStorage from a _Rep32.

        Args:
          rep: A shape representation.
        """
        var rep_copy = rep
        var rep_ptr = UnsafePointer(to=rep_copy)
        self = rep_ptr.bitcast[_TensorShapeStorage]()[]

    @always_inline
    @implicit
    fn __init__(out self, rep: _RepOutOfLine):
        """Initializes the _TensorShapeStorage from a _Rep32.

        Note that this will not copy the underlying data.

        Args:
          rep: A shape representation.
        """
        var rep_copy = rep
        var rep_ptr = UnsafePointer(to=rep_copy)
        self = rep_ptr.bitcast[_TensorShapeStorage]()[]


@always_inline
fn _as_rep16(rep0: _TensorShapeStorage) -> _Rep16:
    """Constructs a _Rep16 representation from a _TensorShapeStorage.

    Args:
      rep0: The _TensorShapeStorage representation.

    Returns:
      The _Rep16 representation.
    """
    var rep = rep0
    return UnsafePointer(to=rep).bitcast[_Rep16]()[]


@always_inline
fn _as_rep16(rep0: _Rep16) -> _Rep16:
    """Constructs a _Rep16 representation from a _Rep16 representation.

    Args:
      rep0: The _Rep16 representation.

    Returns:
      The _Rep16 representation.
    """
    return rep0


@always_inline
fn _as_rep16(rep0: _Rep32) -> _Rep16:
    """Constructs a _Rep16 representation from a _Rep32 representation.

    Args:
      rep0: The _Rep32 representation.

    Returns:
      The _Rep16 representation.
    """
    var rep = rep0
    return UnsafePointer(to=rep).bitcast[_Rep16]()[]


@always_inline
fn _as_rep32(rep0: _TensorShapeStorage) -> _Rep32:
    """Constructs a _Rep32 representation from a _TensorShapeStorage.

    Args:
      rep0: The _TensorShapeStorage representation.

    Returns:
      The _Rep32 representation.
    """
    var rep = rep0
    return UnsafePointer(to=rep).bitcast[_Rep32]()[]


@always_inline
fn _as_rep32(rep0: _Rep16) -> _Rep32:
    """Constructs a _Rep32 representation from a _Rep16 representation.

    Args:
      rep0: The _Rep16 representation.

    Returns:
      The _Rep32 representation.
    """
    var rep = rep0
    return UnsafePointer(to=rep).bitcast[_Rep32]()[]


@always_inline
fn _as_rep32(rep0: _Rep32) -> _Rep32:
    """Constructs a _Rep32 representation from a _Rep32 representation.

    Args:
      rep0: The _Rep32 representation.

    Returns:
      The _Rep32 representation.
    """
    return rep0


@always_inline
fn _as_rep_out_of_line(rep0: _TensorShapeStorage) -> _RepOutOfLine:
    """Constructs a _RepOutOfLine representation from a _TensorShapeStorage.

    Args:
      rep0: The _TensorShapeStorage representation.

    Returns:
      The _RepOutOfLine representation.
    """
    var rep = rep0
    return UnsafePointer(to=rep).bitcast[_RepOutOfLine]()[]


@always_inline
fn _as_rep_out_of_line(rep0: _Rep16) -> _RepOutOfLine:
    """Constructs a _RepOutOfLine representation from a _Rep16 representation.

    Args:
      rep0: The _Rep16 representation.

    Returns:
      The _RepOutOfLine representation.
    """
    var rep = rep0
    return UnsafePointer(to=rep).bitcast[_RepOutOfLine]()[]


@always_inline
fn _as_rep_out_of_line(rep0: _Rep32) -> _RepOutOfLine:
    """Constructs a _RepOutOfLine representation from a _Rep32 representation.

    Args:
      rep0: The _Rep32 representation.

    Returns:
      The _RepOutOfLine representation.
    """
    var rep = rep0
    return UnsafePointer(to=rep).bitcast[_RepOutOfLine]()[]


# ===-----------------------------------------------------------------------===#
# TensorShape
# ===-----------------------------------------------------------------------===#


@value
struct TensorShape(Stringable, Writable, Copyable, Movable, EqualityComparable):
    """A space efficient representation of a tensor shape. This struct
    implements value semantics and owns its underlying data."""

    var _rep: _TensorShapeStorage
    """The underlying _TensorShapeStorage backing."""

    @always_inline
    fn __init__(out self):
        """Default initializer for TensorShape."""
        self._rep = _TensorShapeStorage()

    @always_inline
    @implicit
    fn __init__(out self, rep: _TensorShapeStorage):
        """Default initializer for TensorShape."""
        self._rep = rep

    @always_inline
    @implicit
    fn __init__[*Ts: Copyable & Movable](out self, shapes: Tuple[*Ts]):
        """Initializes a TensorShape from the values provided.

        Args:
          shapes: The shapes to initialize the shape with.
        """

        @parameter
        fn variadic_size(
            x: __mlir_type[`!kgen.variadic<`, Copyable & Movable, `>`]
        ) -> Int:
            return __mlir_op.`pop.variadic.size`(x)

        alias rank = variadic_size(shapes.element_types)
        var tuple = IndexList[rank]()

        @parameter
        for i in range(rank):
            alias T = Ts[i]

            @parameter
            if not _type_is_eq[T, Int]() and not _type_is_eq[T, UInt]():
                constrained[False, "shape should consist of integer values"]()
            tuple[i] = rebind[Int](shapes[i])

        self = Self(tuple)

    @always_inline
    @implicit
    fn __init__(out self, *shapes: Int):
        """Initializes a TensorShape from the values provided.

        Args:
          shapes: The shapes to initialize the shape with.
        """
        self = TensorShape(shapes)

    @always_inline
    @implicit
    fn __init__(out self, shapes: VariadicList[Int]):
        """Initializes a TensorShape from the values provided.

        Args:
          shapes: The shapes to initialize the shape with.
        """
        var rank = len(shapes)

        # Decide which representation we can use and initialize the elements.
        # The most common case should fit into 4 dimensions.
        if rank <= 4:
            var ok = True  # Checks if we did not loose precision.
            var rep = _Rep32()
            rep.rank = rank
            for i in range(rank):
                rep[i] = shapes[i]
                if rep[i] != shapes[i]:
                    ok = False
                    break
            if ok:
                self._rep = rep
                return

        # Otherwise we fall through to try the next representation.
        # Virtually everything else will fit into 6 dimensions.
        if rank <= 6:
            var ok = True  # Checks if we did not loose precision.
            var rep = _Rep16()
            rep.rank = rank
            for i in range(rank):
                rep[i] = shapes[i]
                if rep[i] != shapes[i]:
                    ok = False
                    break
            if ok:
                self._rep = rep
                return

        # Otherwise, we will store out of line.
        var rep = _RepOutOfLine()
        rep.rank = rank
        rep.dims = UnsafePointer[Scalar[DType.index]].alloc(rank)
        for i in range(rank):
            rep[i] = shapes[i]

        self._rep = rep

    @always_inline
    @implicit
    fn __init__(out self, shapes: List[Int, *_]):
        """Initializes a TensorShape from the list provided.

        Args:
          shapes: The list to initialize the shape with.
        """
        var rank = len(shapes)

        # Decide which representation we can use and initialize the elements.
        # The most common case should fit into 4 dimensions.
        if rank <= 4:
            var ok = True  # Checks if we did not loose precision.
            var rep = _Rep32()
            rep.rank = rank
            for i in range(rank):
                rep[i] = shapes[i]
                if rep[i] != shapes[i]:
                    ok = False
                    break
            if ok:
                self._rep = rep
                return

        # Otherwise we fall through to try the next representation.
        # Virtually everything else will fit into 6 dimensions.
        if rank <= 6:
            var ok = True  # Checks if we did not loose precision.
            var rep = _Rep16()
            rep.rank = rank
            for i in range(rank):
                rep[i] = shapes[i]
                if rep[i] != shapes[i]:
                    ok = False
                    break
            if ok:
                self._rep = rep
                return

        # Otherwise, we will store out of line.
        var rep = _RepOutOfLine()
        rep.rank = rank
        rep.dims = UnsafePointer[Scalar[DType.index]].alloc(rank)
        for i in range(rank):
            rep[i] = shapes[i]

        self._rep = rep

    @always_inline
    @implicit
    fn __init__[rank: Int](out self, shapes: IndexList[rank]):
        """Initializes a TensorShape from the values provided.

        Parameters:
            rank: The rank.

        Args:
            shapes: The shapes to initialize the shape with.
        """

        # Decide which representation we can use and initialize the elements.
        # The most common case should fit into 4 dimensions.
        @parameter
        if rank <= 4:
            var ok = True  # Checks if we did not loose precision.
            var rep = _Rep32()
            rep.rank = rank
            for i in range(rank):
                rep[i] = shapes[i]
                if rep[i] != shapes[i]:
                    ok = False
                    break
            if ok:
                self._rep = rep
                return

        # Otherwise we fall through to try the next representation.
        # Virtually everything else will fit into 6 dimensions.
        @parameter
        if rank <= 6:
            var ok = True  # Checks if we did not loose precision.
            var rep = _Rep16()
            rep.rank = rank
            for i in range(rank):
                rep[i] = shapes[i]
                if rep[i] != shapes[i]:
                    ok = False
                    break
            if ok:
                self._rep = rep
                return

        # Otherwise, we will store out of line.
        var rep = _RepOutOfLine()
        rep.rank = rank
        rep.dims = UnsafePointer[Scalar[DType.index]].alloc(rank)
        for i in range(rank):
            rep[i] = shapes[i]

        self._rep = rep

    @always_inline
    fn __copyinit__(out self, other: Self):
        """Creates a deep copy of an existing shape.

        Args:
            other: The shape to copy.
        """
        if other._is_out_of_line():
            # TODO: memcpy the pointer
            self._rep = _TensorShapeStorage(
                _as_rep_out_of_line(other._rep).copy()
            )
        else:
            self._rep = _TensorShapeStorage(_as_rep16(other._rep))

    @always_inline
    fn __del__(owned self):
        """Delete the shape and release any owned memory."""
        if self._is_out_of_line():
            var out_of_line = _as_rep_out_of_line(self._rep)
            out_of_line.dims.free()

    @always_inline
    fn _get_rep_kind(self) -> _RepKind:
        """Gets the underlying representation kind.

        Returns:
          The underlying representation kind.
        """
        return _as_rep32(self._rep).rep_kind

    @always_inline
    fn __getitem__(self, index: Int) -> Int:
        """Gets the dimension at the specified index.

        Args:
          index: The dimension index.

        Returns:
          The dimension at the specified index.
        """
        var normalized_index = self.rank() + index if index < 0 else index

        var rep_kind = self._get_rep_kind()
        if rep_kind == _RepKind.KIND_16:
            return _as_rep16(self._rep)[normalized_index]
        if rep_kind == _RepKind.KIND_32:
            return _as_rep32(self._rep)[normalized_index]
        if rep_kind == _RepKind.KIND_OUT_OF_LINE:
            return _as_rep_out_of_line(self._rep)[normalized_index]
        return -1

    @always_inline
    fn _is_out_of_line(self) -> Bool:
        """Checks if the representation is out of line.

        Returns:
          True if the representation is out of line and False otherwise.
        """
        return self._get_rep_kind() == _RepKind.KIND_OUT_OF_LINE

    @always_inline
    fn __eq__(self, other: Self) -> Bool:
        """Returns True if the two values are the same and False otherwise.

        Args:
          other: The other TensorShape to compare against.

        Returns:
          True if the two shapes are the same and False otherwise.
        """
        if self.rank() != other.rank():
            return False

        # If the two representations match, then we can do a quick check.
        if self._get_rep_kind() == other._get_rep_kind():
            var rep_kind = self._get_rep_kind()
            if rep_kind == _RepKind.KIND_16:
                return _as_rep16(self._rep) == _as_rep16(other._rep)
            if rep_kind == _RepKind.KIND_32:
                return _as_rep32(self._rep) == _as_rep32(other._rep)
            if rep_kind == _RepKind.KIND_OUT_OF_LINE:
                return _as_rep_out_of_line(self._rep) == _as_rep_out_of_line(
                    other._rep
                )

        # Otherwise, we check if the dimentions match.
        for i in range(self.rank()):
            if self[i] != other[i]:
                return False

        return True

    @always_inline
    fn __ne__(self, other: Self) -> Bool:
        """Returns True if the two values are not the same and False otherwise.

        Args:
          other: The other TensorShape to compare against.

        Returns:
          True if the two shapes are the not the same and False otherwise.
        """

        return not (self == other)

    @always_inline
    fn rank(self) -> Int:
        """Gets the rank of the shape.

        Returns:
          The rank of the shape.
        """
        return _as_rep32(self._rep).get_rank()

    @always_inline
    fn num_elements(self) -> Int:
        """Gets the total number of elements in the shape.

        Returns:
          The total number of elements in the shape.
        """
        var rep_kind = self._get_rep_kind()
        if rep_kind == _RepKind.KIND_16:
            return _as_rep16(self._rep).get_num_elements()
        if rep_kind == _RepKind.KIND_32:
            return _as_rep32(self._rep).get_num_elements()
        if rep_kind == _RepKind.KIND_OUT_OF_LINE:
            return _as_rep_out_of_line(self._rep).get_num_elements()
        return -1

    @always_inline
    fn __repr__(self) -> String:
        """Returns the string representation of the shape.

        Returns:
          The string representation of the shape.
        """
        return self.__str__()

    @always_inline
    fn __str__(self) -> String:
        """Returns the string representation of the shape.

        Returns:
          The string representation of the shape.
        """

        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
        """
        Formats this TensorShape to the provided Writer.

        Parameters:
            W: A type conforming to the Writable trait.

        Args:
            writer: The object to write to.
        """

        var rep_kind = self._get_rep_kind()

        if rep_kind == _RepKind.KIND_16:
            return writer.write(_as_rep16(self._rep))
        if rep_kind == _RepKind.KIND_32:
            return writer.write(_as_rep32(self._rep))
        if rep_kind == _RepKind.KIND_OUT_OF_LINE:
            return writer.write(_as_rep_out_of_line(self._rep))

        return writer.write("<unknown representation>")
