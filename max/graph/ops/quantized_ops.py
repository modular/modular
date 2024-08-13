# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Optimized quantized operations."""

from max.graph.ops import custom_ops
from max.graph.graph import Graph
from max.graph.graph_value import GraphValue
from max.graph.quantization import QuantizationEncoding
from max.graph.type import Dim, DType, TensorType


_REPACK_WEIGHTS_CUSTOM_OP_NAMES = {
    QuantizationEncoding.Q4_0: "vroom_q4_0_repack_weights",
    QuantizationEncoding.Q4_K: "vroom_q4_k_repack_weights",
    QuantizationEncoding.Q6_K: "vroom_q6_k_repack_weights",
}

_PACKED_QMATMUL_CUSTOM_OP_NAMES = {
    QuantizationEncoding.Q4_0: "vroom_q4_0_matmul",
    QuantizationEncoding.Q4_K: "vroom_q4_k_matmul",
    QuantizationEncoding.Q6_K: "vroom_q6_k_matmul",
}


def _repack_quantized_weights(
    encoding: QuantizationEncoding, rhs: GraphValue
) -> GraphValue:
    rhs_type = rhs.tensor_type

    op_name = _REPACK_WEIGHTS_CUSTOM_OP_NAMES.get(encoding)
    if op_name is None:
        raise ValueError(f"unsupported quantization encoding {encoding}")

    return custom_ops.custom(
        op_name,
        [rhs],
        out_types=[
            TensorType(
                DType.uint8, (rhs_type.shape[0], rhs_type.shape[1])
            ).to_mlir()
        ],
    )[0]


def _packed_qmatmul(
    encoding: QuantizationEncoding,
    lhs_matrix: GraphValue,
    rhs_repack: GraphValue,
) -> GraphValue:
    op_name = _PACKED_QMATMUL_CUSTOM_OP_NAMES.get(encoding)
    if op_name is None:
        raise ValueError(f"unsupported quantization encoding {encoding}")

    return custom_ops.custom(
        op_name,
        [lhs_matrix, rhs_repack],
        out_types=[
            TensorType(
                DType.float32, (lhs_matrix.shape[0], rhs_repack.shape[0])
            ).to_mlir(),
        ],
    )[0]


def qmatmul(
    encoding: QuantizationEncoding, lhs: GraphValue, rhs: GraphValue
) -> GraphValue:
    """Performs matrix multiplication between floating point and quantized
    tensors.

    This quantizes the `lhs` floating point value to match the encoding of the
    `rhs` quantized value, performs matmul, and then dequantizes the result.
    Beware that, compared to a regular matmul op, this one expects the `rhs`
    value to be transposed. For example, if the `lhs` shape is `[32, 64]`, and
    the quantized `rhs` shape is also `[32, 64]`, then the output shape is
    `[32, 32]`

    That is, this function returns the result from:

        dequantize(quantize(lhs) @ transpose(rhs))

    The last two dimensions in `lhs` are treated as matrices and multiplied
    by `rhs` (which must be a 2D tensor). Any remaining dimensions in `lhs`
    are broadcast dimensions.

    NOTE: Currently this supports Q4_0, Q4_K, and Q6_K encodings only.

    Parameters:
        encoding: The quantization encoding to use.

    Args:
        lhs: The non-quantized, left-hand-side of the matmul.
        rhs: The transposed and quantized right-hand-side of the matmul.
             Must be rank 2 (a 2D tensor/matrix) and in a supported
             [quantization encoding](/max/api/mojo/graph/quantization/).

    Returns:
        The dequantized result (a floating point tensor).
    """
    # Quantized matmul for supported quantized encoding types.
    # rhs is uint8 and in a packed format such as Q4_0, Q4_K, or Q6_K.
    if rhs.tensor_type.dtype is not DType.uint8:
        raise TypeError(f"expected uint8 DType but got {rhs.tensor_type.dtype}")

    if len(rhs.shape) != 2:
        raise TypeError("rhs must be a matrix")

    # Reshape LHS to a matrix, which is expected by the q4_0 matmul op.
    lhs_matrix = lhs.reshape((-1, lhs.shape[-1]))
    # Rebinding here breaks the reshape later, see GRA-881.
    # Fortunately things work without the rebind.
    # prod_dim = Graph.current.unique_symbolic_dim("qmatmul")
    # lhs_matrix = lhs_matrix.rebind((prod_dim, lhs.shape[-1]))

    # Prepack weights.
    rhs_repack = _repack_quantized_weights(encoding, rhs)

    # Perform quantized matmul.
    qmatmul_out = _packed_qmatmul(encoding, lhs_matrix, rhs_repack)

    # Reshape matmul output to restore the original rank(lhs) - 1 dimensions.
    return qmatmul_out.reshape((*lhs.shape[:-1], rhs.shape[0]))
