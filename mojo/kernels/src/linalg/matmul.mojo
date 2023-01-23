# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Buffer import NDBuffer
from Bool import Bool
from Int import Int
from Transpose import _index2D
from SIMD import SIMD


struct GemmShape[
    shape: __mlir_type[`!kgen.list<index[2]>`],
    type: __mlir_type.`!kgen.dtype`,
]:
    """Helper class to unpack gemm dimension and layout."""

    var M: Int
    var N: Int
    var K: Int
    var transposeA: Bool
    var transposeB: Bool

    # Construct from dynamic shaped input.
    fn __new__(
        c: NDBuffer[2, shape, type],
        a: NDBuffer[2, shape, type],
        b: NDBuffer[2, shape, type],
        transposeA: Bool,
        transposeB: Bool,
    ) -> GemmShape[shape, type]:
        """Constructor of a gemm shape record from input buffers.

        Args:
            c: Buffer with allocated output space.
            a: Buffer containing matrix operand A.
            b: Buffer containing matrix operand B.
        """
        var gemmShape: GemmShape[shape, type]
        gemmShape.transposeA = transposeA
        gemmShape.transposeB = transposeB
        gemmShape.M = c.dim[0]()
        gemmShape.N = c.dim[1]()
        if transposeA:
            gemmShape.K = a.dim[0]()
        else:
            gemmShape.K = a.dim[1]()
        return gemmShape


struct _Matrix[
    shape: __mlir_type[`!kgen.list<index[2]>`],
    type: __mlir_type.`!kgen.dtype`,
    transposed: Bool,
]:
    """Utility to access matrix across layouts with
    unified indexing interface.
    """

    var data: NDBuffer[2, shape, type]

    fn __new__(
        data: NDBuffer[2, shape, type]
    ) -> _Matrix[shape, type, transposed]:
        """Constructor of a matrix based on a buffer and a transpose flag.

        Args:
            data: The buffer containing the matrix data.

        Returns:
            The constructed matrix.
        """

        return _Matrix[shape, type, transposed] {data: data}

    fn __getitem__(self, x: Int, y: Int) -> SIMD[1, type]:
        """Returns the data stored at the given untransposed coordinate.

        Args:
            x: The untransposed x coordinate.
            y: The untransposed y coordinate.

        Returns:
            The value stored at the coordinate.
        """
        if transposed:
            return self.data.__getitem__(_index2D(y, x))
        return self.data.__getitem__(_index2D(x, y))

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
            self.data.__setitem__(_index2D(y, x), val)
        else:
            self.data.__setitem__(_index2D(x, y), val)


fn naive_matmul[
    shape: __mlir_type[`!kgen.list<index[2]>`],
    type: __mlir_type.`!kgen.dtype`,
    transposeA: Bool,
    transposeB: Bool,
](
    c: NDBuffer[2, shape, type],
    a: NDBuffer[2, shape, type],
    b: NDBuffer[2, shape, type],
):
    """Computes matrix multiplication with a naive algorithm.

    Args:
        c: Buffer with allocated output space.
        a: Buffer containing matrix operand A.
        b: Buffer containing matrix operand B.
        transposeA: indicates if a is transposed.
        transposeB: indicates if b is transposed.
    """
    var gemmShape = GemmShape[shape, type](c, a, b, transposeA, transposeB)
    var matrixA = _Matrix[shape, type, transposeA](a)
    var matrixB = _Matrix[shape, type, transposeB](b)
    var matrixC = _Matrix[shape, type, False](c)

    var m: Int = 0
    while m < gemmShape.M:
        var n: Int = 0
        while n < gemmShape.N:
            var cVal: SIMD[1, type] = 0
            var k: Int = 0
            while k < gemmShape.K:
                var aVal = matrixA.__getitem__(m, k)
                var bVal = matrixB.__getitem__(k, n)
                cVal += aVal * bVal
                k += 1
            matrixC.__setitem__(m, n, cVal)
            n += 1
        m += 1
