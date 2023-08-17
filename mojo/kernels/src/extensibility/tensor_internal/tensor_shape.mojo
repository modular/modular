# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


from Assert import assert_param, debug_assert
from List import VariadicList
from Index import StaticIntTuple
from Memory import memcpy
from Pointer import Pointer, DTypePointer
from Range import range
from StaticTuple import StaticTuple
from String import String
from sys.info import sizeof, is_little_endian


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


# ===----------------------------------------------------------------------===#
# Utilities
# ===----------------------------------------------------------------------===#


@register_passable("trivial")
struct _RepKind:
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
    fn __init__(value: Int) -> _RepKind:
        return Self {kind: value}

    @always_inline("nodebug")
    fn __eq__(self, rhs: _RepKind) -> Bool:
        return (self.kind == rhs.kind).__bool__()


@register_passable("trivial")
struct _Rep16:
    """A representation which can hold up to 6 dimensions with each dim
    occupying at most 16-bits."""

    var dims: StaticTuple[6, Int16]
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
    fn __init__() -> Self:
        """Default initializes the _Rep16 type."""
        return Self {
            dims: StaticTuple[6, Int16](),
            _unused: 0,
            rep_kind: _RepKind.KIND_16,
            rank: 0,
            auxillary: 0,
        }

    @always_inline
    fn get_rank(self) -> Int:
        """Gets the rank of the representation.

        Returns:
          The rank of the representation.
        """
        debug_assert(self.rank.to_int() < 4, "index out of range")
        return self.rank.to_int()

    @always_inline
    fn get_num_elements(self) -> Int:
        """Gets the number of elements of the representation.

        Returns:
          The number of elements in the representation.
        """
        let rank = self.get_rank()
        var product: Int = 1
        for i in range(rank):
            product *= self.dims[i].to_int()
        return product

    @always_inline
    fn __getitem__(self, index: Int) -> Int:
        """Gets dimension at the specified index.

        Args:
          index: the dimension index.

        Returns:
          The value at the specified dimension.
        """
        return self.dims[index].to_int()

    @always_inline
    fn __setitem__(inout self, index: Int, val: Int):
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
        var buf = String("")
        for i in range(self.get_rank()):
            if i != 0:
                buf += "x"
            buf += String(self[i])
        return buf


@register_passable("trivial")
struct _Rep32:
    """A representation which can hold up to 4 dimensions with the first three
    occupying at most 32-bits and the last occupying at most 8 bits."""

    var dims012: StaticTuple[3, Int32]  # dim0, dim1, dim2
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
    fn __init__() -> Self:
        """Default initializes the _Rep32 type."""
        return Self {
            dims012: StaticTuple[3, Int32](),
            dim3: 0,
            rep_kind: _RepKind.KIND_32,
            rank: 0,
            auxillary: 0,
        }

    @always_inline
    fn get_rank(self) -> Int:
        """Gets the rank of the representation.

        Returns:
          The rank of the representation.
        """
        debug_assert(self.rank.to_int() < 4, "index out of range")
        return self.rank.to_int()

    @always_inline
    fn get_num_elements(self) -> Int:
        """Gets the number of elements of the representation.

        Returns:
          The number of elements in the representation.
        """
        var rank = self.get_rank()
        var product: Int = 1
        if rank == 4:
            product = self.dim3.to_int()
            rank -= 1
        for i in range(rank):
            product *= self.dims012[i].to_int()
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
            return self.dim3.to_int()
        else:
            return self.dims012[index].to_int()

    @always_inline
    fn __setitem__(inout self, index: Int, val: Int):
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
        var buf = String("")
        for i in range(self.get_rank()):
            if i != 0:
                buf += "x"
            buf += String(self[i])
        return buf


@register_passable("trivial")
struct _RepOutOfLine:
    """A general storage kind which stores the dimensions on the heap."""

    alias _padding_size = (
        13 - sizeof[DTypePointer[DType.invalid].pointer_type]()
    )
    var dims: DTypePointer[DType.index]
    """The heap allocated dimensions."""
    # FIXME: This isn't correct for big endian systems, but we check with
    # static_assert below.
    var _padding: StaticTuple[Self._padding_size, UInt8]
    """Unused padding value."""
    var rep_kind: _RepKind
    """The representation kind."""
    var rank: UInt8
    """The rank of the shape."""
    var auxillary: UInt8
    """Auxillary information about the shape."""

    @always_inline
    fn __init__() -> Self:
        """Default initializes the _RepOutOfLine type."""
        assert_param[
            is_little_endian(),
            (
                "the out of line representation is only implemetned on little"
                " endian systems"
            ),
        ]()
        return Self {
            dims: DTypePointer[DType.index](),
            _padding: StaticTuple[Self._padding_size, UInt8](),
            rep_kind: _RepKind.KIND_OUT_OF_LINE,
            rank: 0,
            auxillary: 0,
        }

    @always_inline
    fn get_rank(self) -> Int:
        """Gets the rank of the representation.

        Returns:
          The rank of the representation.
        """
        return self.rank.to_int()

    @always_inline
    fn __getitem__(self, index: Int) -> Int:
        """Gets dimension at the specified index.

        Args:
          index: the dimension index.

        Returns:
          The value at the specified dimension.
        """
        return self.dims.load(index).to_int()

    @always_inline
    fn __setitem__(inout self, index: Int, val: Int):
        """Sets the dimension at the specified index.

        Args:
          index: the dimension index.
          val: the value to set.
        """
        debug_assert(index < self.get_rank(), "index out of range")
        self.dims.store(index, val)

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
        let dims_copy = DTypePointer[DType.index].alloc(self.get_rank())
        memcpy(dims_copy, self.dims, self.get_rank())

        return Self {
            dims: dims_copy,
            _padding: self._padding,
            rep_kind: self.rep_kind,
            rank: self.rank,
            auxillary: self.auxillary,
        }

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
        var buf = String("")
        for i in range(self.get_rank()):
            if i != 0:
                buf += "x"
            buf += String(self[i])
        return buf


@register_passable("trivial")
struct _TensorShapeStorage:
    """The storage type for the tensor shape. This acts as a union type between
    all the representations."""

    var ptr: DTypePointer[DType.invalid]
    var idx: Int64

    @always_inline
    fn __init__() -> Self:
        """Default initializes the _TensorShapeStorage type."""
        var rep = _Rep32()
        let rep_ptr = Pointer.address_of(rep)
        return rep_ptr.bitcast[_TensorShapeStorage]().load()

    @always_inline
    fn __init__(rep: _Rep16) -> Self:
        """Initializes the _TensorShapeStorage from a _Rep16.

        Args:
          rep: A shape representation.

        Returns:
          The _TensorShapeStorage.
        """
        var rep_copy = rep
        let rep_ptr = Pointer.address_of(rep_copy)
        return rep_ptr.bitcast[_TensorShapeStorage]().load()

    @always_inline
    fn __init__(rep: _Rep32) -> Self:
        """Initializes the _TensorShapeStorage from a _Rep32.

        Args:
          rep: A shape representation.

        Returns:
          The _TensorShapeStorage.
        """
        var rep_copy = rep
        let rep_ptr = Pointer.address_of(rep_copy)
        return rep_ptr.bitcast[_TensorShapeStorage]().load()

    @always_inline
    fn __init__(rep: _RepOutOfLine) -> Self:
        """Initializes the _TensorShapeStorage from a _Rep32.

        Note that this will not copy the underlying data.

        Args:
          rep: A shape representation.

        Returns:
          The _TensorShapeStorage.
        """
        var rep_copy = rep
        let rep_ptr = Pointer.address_of(rep_copy)
        return rep_ptr.bitcast[_TensorShapeStorage]().load()


@always_inline
fn _as_rep16(rep0: _TensorShapeStorage) -> _Rep16:
    """Constructs a _Rep16 representation from a _TensorShapeStorage.

    Args:
      rep0: The _TensorShapeStorage representation.

    Returns:
      The _Rep16 representation.
    """
    var rep = rep0
    let rep_ptr = Pointer.address_of(rep)
    return rep_ptr.bitcast[_Rep16]().load()


@always_inline
fn _as_rep16(rep0: _Rep16) -> _Rep16:
    """Constructs a _Rep16 representation from a _Rep16 representation.

    Args:
      rep0: The _Rep16 representation.

    Returns:
      The _Rep16 representation.
    """
    let rep = rep0
    return rep


@always_inline
fn _as_rep16(rep0: _Rep32) -> _Rep16:
    """Constructs a _Rep16 representation from a _Rep32 representation.

    Args:
      rep0: The _Rep32 representation.

    Returns:
      The _Rep16 representation.
    """
    var rep = rep0
    let rep_ptr = Pointer.address_of(rep)
    return rep_ptr.bitcast[_Rep16]().load()


@always_inline
fn _as_rep32(rep0: _TensorShapeStorage) -> _Rep32:
    """Constructs a _Rep32 representation from a _TensorShapeStorage.

    Args:
      rep0: The _TensorShapeStorage representation.

    Returns:
      The _Rep32 representation.
    """
    var rep = rep0
    let rep_ptr = Pointer.address_of(rep)
    return rep_ptr.bitcast[_Rep32]().load()


@always_inline
fn _as_rep32(rep0: _Rep16) -> _Rep32:
    """Constructs a _Rep32 representation from a _Rep16 representation.

    Args:
      rep0: The _Rep16 representation.

    Returns:
      The _Rep32 representation.
    """
    var rep = rep0
    let rep_ptr = Pointer.address_of(rep)
    return rep_ptr.bitcast[_Rep32]().load()


@always_inline
fn _as_rep32(rep0: _Rep32) -> _Rep32:
    """Constructs a _Rep32 representation from a _Rep32 representation.

    Args:
      rep0: The _Rep32 representation.

    Returns:
      The _Rep32 representation.
    """
    let rep = rep0
    return rep


@always_inline
fn _as_rep_out_of_line(rep0: _TensorShapeStorage) -> _RepOutOfLine:
    """Constructs a _RepOutOfLine representation from a _TensorShapeStorage.

    Args:
      rep0: The _TensorShapeStorage representation.

    Returns:
      The _RepOutOfLine representation.
    """
    var rep = rep0
    let rep_ptr = Pointer.address_of(rep)
    return rep_ptr.bitcast[_RepOutOfLine]().load()


@always_inline
fn _as_rep_out_of_line(rep0: _Rep16) -> _RepOutOfLine:
    """Constructs a _RepOutOfLine representation from a _Rep16 representation.

    Args:
      rep0: The _Rep16 representation.

    Returns:
      The _RepOutOfLine representation.
    """
    var rep = rep0
    let rep_ptr = Pointer.address_of(rep)
    return rep_ptr.bitcast[_RepOutOfLine]().load()


@always_inline
fn _as_rep_out_of_line(rep0: _Rep32) -> _RepOutOfLine:
    """Constructs a _RepOutOfLine representation from a _Rep32 representation.

    Args:
      rep0: The _Rep32 representation.

    Returns:
      The _RepOutOfLine representation.
    """
    var rep = rep0
    let rep_ptr = Pointer.address_of(rep)
    return rep_ptr.bitcast[_RepOutOfLine]().load()


# ===----------------------------------------------------------------------===#
# TensorShape
# ===----------------------------------------------------------------------===#


struct TensorShape:
    """A space efficient representation of a tensor shape. This struct
    implements value semantics and owns its underlying data."""

    var _rep: _TensorShapeStorage
    """The underlying _TensorShapeStorage backing."""

    @always_inline
    fn __init__(inout self):
        """Default initializer for TensorShape."""
        self._rep = _TensorShapeStorage()

    @always_inline
    fn __init__(inout self, *shapes: Int):
        """Initializes a TensorShape from the values provided.

        Args:
          shapes: The shapes to initialize the shape with.
        """
        self = TensorShape(VariadicList[Int](shapes))

    @always_inline
    fn __init__(inout self, shapes: VariadicList[Int]):
        """Initializes a TensorShape from the values provided.

        Args:
          shapes: The shapes to initialize the shape with.
        """
        let rank = shapes.__len__()

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
        rep.dims = DTypePointer[DType.index].alloc(rank)
        for i in range(rank):
            rep[i] = shapes[i]

        self._rep = rep

    @always_inline
    fn __copyinit__(inout self, other: Self):
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
    fn __moveinit__(inout self, owned existing: Self):
        """Move initializer for the shape.

        Args:
            existing: The shape to move.
        """
        self._rep = existing._rep
        existing._rep = _TensorShapeStorage()

    @always_inline
    fn __del__(owned self):
        """Delete the shape and release any owned memory."""
        let rep_kind = self._get_rep_kind()
        if self._is_out_of_line():
            let out_of_line = _as_rep_out_of_line(self._rep)
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
        let rep_kind = self._get_rep_kind()
        if rep_kind == _RepKind.KIND_16:
            return _as_rep16(self._rep)[index]
        if rep_kind == _RepKind.KIND_32:
            return _as_rep32(self._rep)[index]
        if rep_kind == _RepKind.KIND_OUT_OF_LINE:
            return _as_rep_out_of_line(self._rep)[index]
        return -1

    @always_inline
    fn _is_out_of_line(self) -> Bool:
        """Checks if the representation is out of line.

        Returns:
          True if the representation is out of line and False otherwise.
        """
        return self._get_rep_kind() == _RepKind.KIND_OUT_OF_LINE

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
        let rep_kind = self._get_rep_kind()
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
        let rep_kind = self._get_rep_kind()
        if rep_kind == _RepKind.KIND_16:
            return _as_rep16(self._rep).__str__()
        if rep_kind == _RepKind.KIND_32:
            return _as_rep32(self._rep).__str__()
        if rep_kind == _RepKind.KIND_OUT_OF_LINE:
            return _as_rep_out_of_line(self._rep).__str__()
        return "<unknown representation>"
