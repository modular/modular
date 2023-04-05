# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Assert import assert_param_bool
from Buffer import NDBuffer
from DType import DType
from List import DimList
from Index import StaticIntTuple
from Pointer import Pointer, DTypePointer
from SIMD import SIMD


struct Matrix[
    shape: DimList[2],
    type: DType,
    transposed: Bool,
]:
    """Utility to access matrix across layouts with
    unified indexing interface.
    """

    var data: NDBuffer[2, shape, type]

    fn __copy__(self) -> Self:
        return Self {data: self.data}

    fn __init__(self&, data: NDBuffer[2, shape, type]):
        """Constructor of a matrix based on a buffer and a transpose flag.

        Args:
            data: The buffer containing the matrix data.

        Returns:
            The constructed matrix.
        """

        self.data = data

    fn __init__(self&, dptr: DTypePointer[type]):
        """Constructor of a matrix from a DTypePointer.

        Args:
            data: The buffer containing the matrix data.

        Returns:
            The constructed matrix.
        """
        self.data = dptr.address

    fn __init__(
        ptr: Pointer[__mlir_type[`!pop.scalar<`, type.value, `>`]]
    ) -> Matrix[shape, type, transposed]:
        """Constructor of a matrix from a Pointer.

        Args:
            data: The buffer containing the matrix data.

        Returns:
            The constructed matrix.
        """
        let dptr = DTypePointer[type](ptr.address)
        return NDBuffer[2, shape, type](dptr.address)

    fn __getitem__(self, x: Int, y: Int) -> SIMD[1, type]:
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

    fn __setitem__(self, x: Int, y: Int, val: SIMD[1, type]):
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

    fn simd_load[width: Int](self, idxi: Int, idxj: Int) -> SIMD[width, type]:
        assert_param_bool[not transposed]()
        return self.data.simd_load[width](StaticIntTuple[2](idxi, idxj))

    fn simd_store[
        width: Int
    ](self, idxi: Int, idxj: Int, val: SIMD[width, type]):
        assert_param_bool[not transposed]()
        return self.data.simd_store[width](StaticIntTuple[2](idxi, idxj), val)
