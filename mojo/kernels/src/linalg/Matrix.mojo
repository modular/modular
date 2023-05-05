# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Assert import assert_param
from Buffer import NDBuffer
from DType import DType
from List import DimList
from Index import StaticIntTuple
from Pointer import Pointer, DTypePointer
from SIMD import SIMD
from TargetInfo import dtype_alignof


@value
struct Matrix[
    shape: DimList,
    type: DType,
    transposed: Bool,
]:
    """Utility to access matrix across layouts with
    unified indexing interface.
    """

    var data: NDBuffer[2, shape, type]

    fn __init__(inout self, dptr: DTypePointer[type]):
        """Constructor of a matrix from a DTypePointer.

        Args:
            data: The buffer containing the matrix data.

        Returns:
            The constructed matrix.
        """
        self.data = dptr.address

    fn __init__(
        inout self, ptr: Pointer[__mlir_type[`!pop.scalar<`, type.value, `>`]]
    ):
        """Constructor of a matrix from a Pointer.

        Args:
            data: The buffer containing the matrix data.

        Returns:
            The constructed matrix.
        """
        let dptr = DTypePointer[type](ptr.address)
        self = Self(NDBuffer[2, shape, type](dptr.address))

    @staticmethod
    @always_inline
    fn aligned_stack_allocation[
        alignment: Int
    ]() -> Matrix[shape, type, transposed]:
        """Constructs a matrix instance backed by stack allocated memory space.

        Parameters:
            alignment: address alignment requirement for the allocation.

        Returns:
            Constructed matrix with the allocated space.
        """
        return Self(
            NDBuffer[2, shape, type].aligned_stack_allocation[alignment]()
        )

    @staticmethod
    @always_inline
    fn stack_allocation[]() -> Matrix[shape, type, transposed]:
        """Constructs a matrix instance backed by stack allocated memory space.

        Returns:
            Constructed matrix with the allocated space.
        """
        return Matrix[shape, type, transposed].aligned_stack_allocation[
            dtype_alignof[type]()
        ]()

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

        Returns:
            The value stored at the coordinate.
        """
        if transposed:
            self.data[StaticIntTuple[2](y, x)] = val
        else:
            self.data[StaticIntTuple[2](x, y)] = val

    fn simd_load[width: Int](self, idxi: Int, idxj: Int) -> SIMD[type, width]:
        assert_param[not transposed]()
        return self.data.simd_load[width](StaticIntTuple[2](idxi, idxj))

    fn simd_store[
        width: Int
    ](self, idxi: Int, idxj: Int, val: SIMD[type, width]):
        assert_param[not transposed]()
        return self.data.simd_store[width](StaticIntTuple[2](idxi, idxj), val)
