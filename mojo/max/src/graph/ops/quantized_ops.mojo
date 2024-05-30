# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Optimized quantized operators."""

from max.graph.quantization import QuantizationEncoding, Q4_0Encoding

from .custom_ops import custom


def _q4_0_matmul(lhs: Symbol, rhs: Symbol) -> Symbol:
    # Quantized matmul for Q4_0.
    # rhs has uint8 dtype and is in GGML Q4_0 packed format.
    var rhs_dtype = rhs.type().tensor().dtype
    if rhs_dtype != DType.uint8:
        raise "expected uint8 DType but got " + str(rhs_dtype)

    var g = lhs.graph()
    var lhs_type = lhs.tensor_type()
    var rhs_type = rhs.tensor_type()
    if rhs_type.rank() != 2:
        raise "rhs must be a matrix"

    # Compute shapes.
    var lhs_shape = ops.shape_of(lhs)
    var rhs_shape = ops.shape_of(rhs)
    var last_lhs_axis = lhs_type.rank() - 1
    var reshape_shape = ops.stack(
        List(g.scalar(Int64(-1)), lhs_shape[last_lhs_axis])
    )
    var final_shape = ops.concat(
        List(lhs_shape[:last_lhs_axis], rhs_shape[0:1])
    )

    # Compute dims for reshape and matmul result types.
    var final_dims = List[Dim]()
    for i in range(lhs_type.rank() - 1):
        final_dims.append(lhs_type.dim(i))
    final_dims.append(rhs_type.dim(0))

    var matmul_dims = List[Dim]()
    matmul_dims.append(Dim.dynamic())
    matmul_dims.append(lhs_type.dim(-1))

    # Reshape LHS to a matrix, which is expected by the q4_0 matmul op.
    var lhs_matrix = ops.reshape(lhs, reshape_shape, matmul_dims)

    # Prepack weights.
    var rhs_repack = ops.custom["vroom_q4_0_repack_weights"](
        List[Symbol](rhs),
        TensorType(DType.uint8, rhs_type.dim(0), rhs_type.dim(1)),
    )

    # Perform quantized matmul.
    var qmatmul_out = ops.custom["vroom_q4_0_matmul"](
        List[Symbol](lhs_matrix, rhs_repack),
        TensorType(DType.float32, Dim.dynamic(), rhs_type.dim(0)),
    )

    # Reshape matmul output to restore the original rank(lhs) - 1 dimensions.
    return ops.reshape(qmatmul_out, final_shape, final_dims)


def qmatmul[encoding: QuantizationEncoding](lhs: Symbol, rhs: Symbol) -> Symbol:
    """Performs matrix multiplication between floating point and quantized
    tensors.

    This quantizes the `lhs` floating point value to match the encoding of the
    `rhs` quantized value, performs matmul, and then dequantizes the result.
    That is, where `.` is the matmul operator, this function returns the result
    from:

        dequantize(quantize(lhs) . rhs)

    The last two dimensions in `lhs` are treated as matricies and multiplied
    by `rhs` (which must be a 2D tensor). Any remaining dimensions in `lhs`
    are broadcast dimensions.

    Parameters:
        encoding: The quantization encoding to use.

    Args:
        lhs: The non-quantized, left-hand-side of the matmul.
        rhs: The quantized, right-hand-side of the matmul.
             Must be rank 2 (a 2D tensor/matrix) and in a supported
             [quantization encoding](/max/reference/mojo/graph/quantization/).

    Returns:
        The dequantized result (a floating point tensor).
    """

    if encoding.id() == Q4_0Encoding.id():
        return _q4_0_matmul(lhs, rhs)

    raise "unknown quantization encoding in qmatmul: " + encoding.id()
