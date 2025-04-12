# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""
Implements the `TensorSpec` type - a space efficient representation of a
tensor shape and dtype.

You can import these APIs from the `max.tensor` package. For example:

```mojo
from max.tensor import TensorSpec
```
"""

from collections import List
from sys import sizeof

from memory import UnsafePointer

from utils import IndexList, product

from .tensor_shape import TensorShape, _as_rep16

# ===-----------------------------------------------------------------------===#
# TensorSpec
# ===-----------------------------------------------------------------------===#


@value
struct TensorSpec(
    Stringable,
    Writable,
    CollectionElement,
    EqualityComparable,
):
    """A space efficient representation of a tensor shape and dtype. This struct
    implements value semantics and owns its underlying data."""

    var shape: TensorShape
    """The underlying shape of the specification."""

    @always_inline
    fn __init__(out self):
        """Default initializer for TensorShape."""
        self.shape = TensorShape()

    @always_inline
    @implicit
    fn __init__(out self, owned shape: TensorShape):
        self.shape = shape^

    @always_inline
    fn __init__(out self, type: DType, *shapes: Int):
        """Initializes a Tensorspec from the dtype and shapes provided.

        Args:
          type: The dtype of the specification.
          shapes: The shapes to initialize the shape with.
        """
        self = TensorSpec(type, shapes)

    @always_inline
    fn __init__(out self, type: DType, shape: Tuple):
        """Initializes a Tensorspec from the dtype and shapes provided.

        Args:
          type: The dtype of the specification.
          shape: The shapes to initialize the shape with.
        """
        self = TensorSpec(type, TensorShape(shape))

    @always_inline
    fn __init__(out self, type: DType, shapes: VariadicList[Int]):
        """Initializes a Tensorspec from the dtype and shapes provided.

        Args:
          type: The dtype of the specification.
          shapes: The shapes to initialize the shape with.
        """
        self = TensorSpec(type, TensorShape(shapes))

    @always_inline
    fn __init__(out self, type: DType, shapes: List[Int, *_]):
        """Initializes a Tensorspec from the dtype and shapes provided.

        Args:
          type: The dtype of the specification.
          shapes: The shapes to initialize the shape with.
        """
        self = TensorSpec(type, shape=shapes)

    @always_inline
    fn __init__(out self, spec: RuntimeTensorSpec):
        """Initializes a Tensorspec from the RuntimeTensorSpec provided.

        Args:
          spec: The RuntimeTensorSpec to initialize the spec with.
        """
        alias rank = spec.rank
        var shapes = List[Int, hint_trivial_type=True]()
        shapes.reserve(rank)
        for i in range(rank):
            shapes.append(spec[i])
        self = TensorSpec(spec.type, shapes)

    @always_inline
    fn __init__(out self, type: DType, owned shape: TensorShape):
        """Initializes a Tensorspec from the dtype and shape provided.

        Args:
          type: The dtype of the specification.
          shape: The shapes to initialize the shape with.
        """
        var owned_shape = shape^
        var rep = _as_rep16(owned_shape._rep)
        # Set to null so dims won't get freed for RepOutOfLine.
        owned_shape._rep.ptr = UnsafePointer[NoneType]()
        rep.auxillary = type._as_i8()

        self.shape = TensorShape()
        self.shape._rep = rep

    @always_inline
    fn __eq__(self, other: Self) -> Bool:
        """Returns True if the two values are the same and False otherwise.

        Args:
          other: The other TensorSpec to compare against.

        Returns:
          True if the two specs are the same and False otherwise.
        """

        return (
            _as_rep16(self.shape._rep).auxillary
            == _as_rep16(other.shape._rep).auxillary
            and self.shape == other.shape
        )

    @always_inline
    fn __ne__(self, other: Self) -> Bool:
        """Returns True if the two values are not the same and False otherwise.

        Args:
          other: The other TensorSpec to compare against.

        Returns:
          True if the two specs are the not the same and False otherwise.
        """

        return not (self == other)

    @always_inline
    fn __getitem__(self, index: Int) -> Int:
        """Gets the dimension at the specified index.

        Args:
          index: The dimension index.

        Returns:
          The dimension at the specified index.
        """
        return self.shape[index]

    @always_inline
    fn rank(self) -> Int:
        """Gets the rank of the spec.

        Returns:
          The rank of the spec.
        """
        return self.shape.rank()

    @always_inline
    fn dtype(self) -> DType:
        """Gets the DType of the spec.

        Returns:
          The DType of the spec.
        """
        return DType._from_ui8(_as_rep16(self.shape._rep).auxillary.value)

    @always_inline
    fn num_elements(self) -> Int:
        """Gets the total number of elements in the spec.

        Returns:
          The total number of elements in the spec.
        """
        return self.shape.num_elements()

    @always_inline
    fn bytecount(self) -> Int:
        """Gets the total byte count.

        Returns:
          The total byte count.
        """
        return self.num_elements() * self.dtype().sizeof()

    @always_inline
    fn __repr__(self) -> String:
        """Returns the string representation of the spec.

        Returns:
          The string representation of the spec.
        """
        return self.__str__()

    @always_inline
    fn __str__(self) -> String:
        """Returns the string representation of the spec.

        Returns:
          The string representation of the spec.
        """
        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
        """
        Formats this TensorSpec to the provided Writer.

        Parameters:
            W: A type conforming to the Writable trait.

        Args:
            writer: The object to write to.
        """

        writer.write(self.shape, "x", self.dtype())

    @staticmethod
    fn from_bytes(data: UnsafePointer[UInt8]) -> TensorSpec:
        """Create a TensorSpec object from serialized bytes.

        Args:
          data: UnsafePointer to serialized bytes.

        Returns:
          Given bytes as TensorSpec.
        """
        var ptr = data.bitcast[Self]()
        return ptr.take_pointee()


@value
@register_passable("trivial")
struct RuntimeTensorSpec[type: DType, rank: Int]:
    var shape: IndexList[rank]

    @implicit
    fn __init__(out self, spec: TensorSpec):
        """Construct from TensorSpec.

        Args:
            spec: TensorSpec of given rank and type.
        """
        debug_assert(spec.rank() == rank, "rank mismatch")
        debug_assert(spec.dtype() == type, "dtype mismatch")
        self.shape = IndexList[rank]()

        for i in range(rank):
            self.shape[i] = spec[i]

    fn __getitem__(self, idx: Int) -> Int:
        return self.shape[idx]

    fn bytecount(self) -> Int:
        """
        Gets the total byte count.

        Returns:
          The total byte count.
        """
        return product(self.shape) * sizeof[type]()
