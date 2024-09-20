# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""
Implements the `TensorSpec` type.

You can import these APIs from the `max.tensor` package. For example:

```mojo
from max.tensor import TensorSpec
```
"""

from collections import List
from memory import UnsafePointer
from utils import StaticIntTuple
from .tensor_shape import TensorShape, _as_rep16

# ===----------------------------------------------------------------------===#
# TensorSpec
# ===----------------------------------------------------------------------===#


struct TensorSpec(
    Stringable,
    Formattable,
    CollectionElement,
    EqualityComparable,
):
    """A space efficient representation of a tensor shape and dtype. This struct
    implements value semantics and owns its underlying data."""

    var shape: TensorShape
    """The underlying shape of the specification."""

    @always_inline
    fn __init__(inout self):
        """Default initializer for TensorShape."""
        self.shape = TensorShape()

    @always_inline
    fn __init__(inout self, type: DType, *shapes: Int):
        """Initializes a Tensorspec from the dtype and shapes provided.

        Args:
          type: The dtype of the specification.
          shapes: The shapes to initialize the shape with.
        """
        self = TensorSpec(type, shapes)

    @always_inline
    fn __init__(inout self, type: DType, shape: Tuple):
        """Initializes a Tensorspec from the dtype and shapes provided.

        Args:
          type: The dtype of the specification.
          shape: The shapes to initialize the shape with.
        """
        self = TensorSpec(type, TensorShape(shape))

    @always_inline
    fn __init__(inout self, type: DType, shapes: VariadicList[Int]):
        """Initializes a Tensorspec from the dtype and shapes provided.

        Args:
          type: The dtype of the specification.
          shapes: The shapes to initialize the shape with.
        """
        self = TensorSpec(type, TensorShape(shapes))

    @always_inline
    fn __init__(inout self, type: DType, shapes: List[Int, *_]):
        """Initializes a Tensorspec from the dtype and shapes provided.

        Args:
          type: The dtype of the specification.
          shapes: The shapes to initialize the shape with.
        """
        self = TensorSpec(type, shape=shapes)

    @always_inline
    fn __init__(inout self, type: DType, owned shape: TensorShape):
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
    fn __copyinit__(inout self, other: Self):
        """Creates a deep copy of an existing spec.

        Args:
            other: The spec to copy.
        """
        self.shape = other.shape

    @always_inline
    fn __moveinit__(inout self, owned existing: Self):
        """Move initializer for the spec.

        Args:
            existing: The spec to move.
        """
        self.shape = existing.shape^
        existing.shape = TensorShape()

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
        return String.format_sequence(self)

    fn format_to(self, inout writer: Formatter):
        """
        Formats this TensorSpec to the provided formatter.

        Args:
            writer: The formatter to write to.
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
@register_passable
struct StaticTensorSpec[type: DType, rank: Int]:
    var shape: StaticIntTuple[rank]

    fn __init__(inout self, spec: TensorSpec):
        """Construct from TensorSpec.

        Args:
            spec: TensorSpec of given rank and type.
        """
        debug_assert(spec.rank() == rank, "rank mismatch")
        debug_assert(spec.dtype() == type, "dtype mismatch")
        self.shape = StaticIntTuple[rank]()

        for i in range(rank):
            self.shape[i] = spec[i]

    fn get_tensor_spec(self) -> TensorSpec:
        var shapes = List[Int, hint_trivial_type=True]()
        shapes.reserve(rank)
        for i in range(rank):
            shapes.append(self[i])
        return TensorSpec(type, shapes)

    fn __getitem__(self, idx: Int) -> Int:
        return self.shape[idx]
