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
"""Unified layout system for mixed compile-time and runtime indices."""

from builtin.variadics import VariadicOf, VariadicPack
from memory import UnsafePointer


trait MixedIntTupleLike(Copyable, Movable):
    """Trait for unified layout handling of compile-time and runtime indices."""

    pass


@register_passable("trivial")
struct ComptimeInt[value: Int](MixedIntTupleLike):
    """Compile-time known index value.

    Parameters:
        value: The compile-time integer value.
    """

    fn __init__(out self):
        """Initialize a compile-time integer with the specified value."""
        pass


@register_passable("trivial")
struct RuntimeInt[dtype: DType = DType.index](MixedIntTupleLike):
    """Runtime index value with configurable precision.

    Parameters:
        dtype: The data type for the runtime integer value. Defaults to `DType.index`.
    """

    var value: Scalar[dtype]
    """The runtime scalar value."""

    fn __init__(out self, value: Scalar[dtype]):
        """Initialize a runtime integer with the given value.

        Args:
            value: The scalar value to store.
        """
        self.value = value


fn Idx(value: Int) -> RuntimeInt[DType.index]:
    """Helper to create runtime indices.

    Args:
        value: The integer value for the runtime index.

    Returns:
        A `RuntimeInt` instance with the specified value.

    Usage: Idx(5) creates a RuntimeInt with value 5.
    """
    return RuntimeInt[DType.index](value)


fn Idx[value: Int]() -> ComptimeInt[value]:
    """Helper to create compile-time indices.

    Parameters:
        value: The compile-time integer value.

    Returns:
        A `ComptimeInt` instance with the specified compile-time value.

    Usage: Idx[5]() creates a ComptimeInt with value 5.
    """
    return ComptimeInt[value]()


struct MixedIntTuple[*element_types: MixedIntTupleLike](
    MixedIntTupleLike, Sized
):
    """A struct representing tuple-like data with compile-time and runtime elements.

    Parameters:
        element_types: The variadic pack of element types that implement `MixedIntTupleLike`.
    """

    # TODO(MOCO-1565): Use a Tuple[*element_types] instead of directly using a variadic pack,
    # therefore eliminating most of the code below.

    alias _mlir_type = __mlir_type[
        `!kgen.pack<:`,
        VariadicOf[MixedIntTupleLike],
        element_types,
        `>`,
    ]

    var storage: Self._mlir_type
    """The underlying MLIR storage for the tuple elements."""

    @staticmethod
    fn __len__() -> Int:
        """Get the length of the tuple.

        Returns:
            The number of elements in the tuple.
        """

        @parameter
        fn variadic_size(x: VariadicOf[MixedIntTupleLike]) -> Int:
            return __mlir_op.`pop.variadic.size`(x)

        return variadic_size(element_types)

    fn __len__(self) -> Int:
        """Get the length of the tuple.

        Returns:
            The number of elements in the tuple.
        """
        return Self.__len__()

    @always_inline("nodebug")
    fn __init__(out self, owned *args: *element_types, __list_literal__: ()):
        """Construct tuple from variadic arguments.

        Args:
            args: Values for each element.
            __list_literal__: List literal marker (unused).
        """
        self = Self(storage=args^)

    @always_inline("nodebug")
    fn __init__(out self, owned *args: *element_types):
        """Construct the tuple from variadic arguments.

        Args:
            args: Initial values for each element.
        """
        self = Self(storage=args^)

    @always_inline("nodebug")
    fn __init__(
        out self,
        *,
        owned storage: VariadicPack[_, _, MixedIntTupleLike, *element_types],
    ):
        """Construct from a low-level variadic pack.

        Args:
            storage: The variadic pack storage to construct from.
        """
        # Mark storage as initialized
        __mlir_op.`lit.ownership.mark_initialized`(
            __get_mvalue_as_litref(self.storage)
        )

        # Move each element into the tuple storage
        @parameter
        for i in range(Self.__len__()):
            UnsafePointer(to=storage[i]).move_pointee_into(
                UnsafePointer(to=self[i])
            )

        # Don't destroy elements when storage goes away
        __disable_del storage

    fn __del__(owned self):
        """Destructor that destroys all elements."""

        @parameter
        for i in range(Self.__len__()):
            UnsafePointer(to=self[i]).destroy_pointee()

    @always_inline("nodebug")
    fn __getitem__[idx: Int](ref self) -> ref [self] element_types[idx.value]:
        """Get a reference to an element in the tuple.

        Parameters:
            idx: The element index to access.

        Returns:
            A reference to the specified element.
        """
        var storage_ptr = UnsafePointer(to=self.storage).address

        # Get pointer to the element
        var elt_ptr = __mlir_op.`kgen.pack.gep`[index = idx.value](storage_ptr)
        # Return as reference, propagating mutability
        return UnsafePointer(elt_ptr)[]
