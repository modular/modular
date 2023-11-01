# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from sys.info import alignof

from memory.buffer import NDBuffer
from memory.unsafe import DTypePointer, Pointer, AddressSpace

from utils.index import StaticIntTuple
from utils.list import DimList


@value
struct Matrix[
    shape: DimList,
    type: DType,
    transposed: Bool,
    /,
    address_space: AddressSpace = AddressSpace.GENERIC,
]:
    """Utility to access matrix across layouts with
    unified indexing interface.
    """

    var data: NDBuffer[2, shape, type, address_space=address_space]

    fn __init__(inout self):
        """Default construct the matrix."""
        self.data = NDBuffer[2, shape, type, address_space=address_space]()

    fn __init__(inout self, dptr: DTypePointer[type, address_space]):
        """Constructor of a matrix from a DTypePointer.

        Args:
            dptr: The buffer containing the matrix data.
        """
        self.data = dptr

    fn __init__(
        inout self,
        dptr: DTypePointer[type, address_space],
        dynamic_shape: StaticIntTuple[2],
    ):
        """Constructs of a matrix from a DTypePointer with dynamic shapes and type.

        Args:
            dptr: Pointer to the data.
            dynamic_shape: A static tuple of size 2 representing shapes.
        """
        self.data = NDBuffer[2, shape, type, address_space=address_space](
            dptr, dynamic_shape
        )

    fn __init__(inout self, ptr: Pointer[SIMD[type, 1], address_space]):
        """Constructor of a matrix from a Pointer.

        Args:
            ptr: The buffer containing the matrix data.
        """
        self.data = NDBuffer[2, shape, type, address_space=address_space](ptr)

    fn __init__(
        inout self,
        ptr: Pointer[SIMD[type, 1], address_space],
        dynamic_shape: StaticIntTuple[2],
    ):
        """Constructs of a matrix from a DTypePointer with dynamic shapes and type.

        Args:
            ptr: Pointer to the data.
            dynamic_shape: A static tuple of size 2 representing shapes.
        """
        self.data = NDBuffer[2, shape, type, address_space=address_space](
            ptr, dynamic_shape
        )

    @staticmethod
    @always_inline
    fn aligned_stack_allocation[alignment: Int]() -> Self:
        """Constructs a matrix instance backed by stack allocated memory space.

        Parameters:
            alignment: Address alignment requirement for the allocation.

        Returns:
            Constructed matrix with the allocated space.
        """
        var res = Self()
        res.data = NDBuffer[
            2, shape, type, address_space=address_space
        ].aligned_stack_allocation[alignment]()
        return res

    @staticmethod
    @always_inline
    fn stack_allocation[]() -> Self:
        """Constructs a matrix instance backed by stack allocated memory space.

        Returns:
            Constructed matrix with the allocated space.
        """
        return Self.aligned_stack_allocation[alignof[type]()]()

    fn __getitem__(self, x: Int, y: Int) -> SIMD[type, 1]:
        """Returns the data stored at the given untransposed coordinate.

        Args:
            x: The untransposed x coordinate.
            y: The untransposed y coordinate.

        Returns:
            The value stored at the coordinate.
        """
        if transposed:
            return self.data[y, x]
        return self.data[x, y]

    fn __setitem__(self, x: Int, y: Int, val: SIMD[type, 1]):
        """Stores the data stored at the given untransposed coordinate.

        Args:
            x: The untransposed x coordinate.
            y: The untransposed y coordinate.
            val: The value to store.
        """
        if transposed:
            self.data[StaticIntTuple[2](y, x)] = val
        else:
            self.data[StaticIntTuple[2](x, y)] = val

    fn simd_load[width: Int](self, idxi: Int, idxj: Int) -> SIMD[type, width]:
        constrained[not transposed]()
        return self.data.simd_load[width](StaticIntTuple[2](idxi, idxj))

    fn simd_store[
        width: Int
    ](self, idxi: Int, idxj: Int, val: SIMD[type, width]):
        constrained[not transposed]()
        return self.data.simd_store[width](StaticIntTuple[2](idxi, idxj), val)
