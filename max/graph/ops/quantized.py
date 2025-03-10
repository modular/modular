# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Optimized quantized operations."""

from typing import Callable, Dict, Literal, Optional, Tuple, Union

from max.dtype import DType

from ..quantization import QuantizationConfig, QuantizationEncoding
from ..type import StaticDim, TensorType
from ..value import TensorValue
from .custom import custom


def repack_gguf_quantized_weights(
    weight: TensorValue,
    quantization_encoding: QuantizationEncoding,
) -> TensorValue:
    quantization_encoding_str = quantization_encoding.name
    return _repack_quantized_weights(
        f"vroom_{quantization_encoding_str}_repack_weights",
        (weight,),
        "vroom",
    )


def _repack_quantized_weights(
    op_name: str,
    rhs: Tuple[TensorValue, ...],
    mode: Literal["gptq", "vroom"],
) -> TensorValue:
    rhs_type = rhs[0].type
    return custom(
        op_name,
        list(rhs),
        out_types=[
            TensorType(
                DType.uint8,
                (
                    (rhs_type.shape[1], rhs_type.shape[0])
                    if mode == "gptq"
                    else (rhs_type.shape[0], rhs_type.shape[1])
                ),
            )
        ],
    )[0].tensor


MODE_TO_DTYPE = {"gptq": DType.bfloat16, "vroom": DType.float32}


def _packed_qmatmul(
    op_name: str,
    lhs_matrix: TensorValue,
    rhs_repack: TensorValue,
    mode: Literal["gptq", "vroom"],
) -> TensorValue:
    return custom(
        op_name,
        [lhs_matrix, rhs_repack],
        out_types=[
            TensorType(
                MODE_TO_DTYPE[mode], (lhs_matrix.shape[0], rhs_repack.shape[0])
            ),
        ],
    )[0].tensor


def _repack_then_matmul(
    repack_op_name: str,
    matmul_op_name: str,
    mode: Literal["gptq", "vroom"],
) -> Callable[[TensorValue, Tuple[TensorValue, ...]], TensorValue]:
    def impl(lhs: TensorValue, rhs: Tuple[TensorValue, ...]) -> TensorValue:
        # Quantized matmul for supported quantized encoding types.
        # rhs is uint8 and in a packed format such as Q4_0, Q4_K, or Q6_K.
        if rhs[0].dtype is not DType.uint8:
            raise TypeError(f"Right-hand side must be uint8, but got {rhs[0]=}")
        dtype = MODE_TO_DTYPE[mode]
        if lhs.dtype is not dtype:
            raise TypeError(
                f"Left-hand side must be {dtype.name}, but got {lhs=}"
            )

        if len(rhs[0].shape) != 2:
            raise TypeError(
                f"Right-hand side must be a matrix, but got {rhs[0]=}"
            )

        lhs_matrix = lhs
        # Reshape LHS to a matrix, which is expected by the q4_0 matmul op.
        if len(lhs.shape) != 2:
            lhs_matrix = lhs_matrix.reshape((-1, lhs.shape[-1]))
        # TODO(MSDK-775): Rebinding here breaks the reshape later.
        # Fortunately things work without the rebind.
        # prod_dim = Graph.current.unique_symbolic_dim("qmatmul")
        # lhs_matrix = lhs_matrix.rebind((prod_dim, lhs.shape[-1]))

        # Prepack weights.
        rhs_repack = _repack_quantized_weights(repack_op_name, rhs, mode)

        # Perform quantized matmul.
        qmatmul_out = _packed_qmatmul(
            matmul_op_name, lhs_matrix, rhs_repack, mode
        )

        if mode == "vroom":
            # Reshape matmul output to restore the original rank(lhs) - 1 dimensions.
            if len(lhs.shape) != 2:
                qmatmul_out = qmatmul_out.reshape(
                    (*lhs.shape[:-1], rhs[0].shape[0])
                )
            return qmatmul_out
        elif mode == "gptq":
            if len(lhs.shape) != 2:
                qmatmul_out = qmatmul_out.reshape(
                    (*lhs.shape[:-1], rhs_repack.shape[0])
                )
            return qmatmul_out
        else:
            assert False

    return impl


# We do not know for sure that all future quantization encodings will best be
# served by the "repack and then matmul" scheme, so this design lets us better
# support future alternative schemes while continuing to support the current
# scheme.
_QMATMUL_STRATEGIES: Dict[
    Union[QuantizationEncoding, str],
    Callable[[TensorValue, Tuple[TensorValue, ...]], TensorValue],
] = {
    "gptq_b4_g128_aTrue": _repack_then_matmul(
        "GPTQ_gpu_repack_b4_g128_desc_act",
        "qmatmul_b4_g128",
        "gptq",
    ),
    "gptq_b4_g128_aFalse": _repack_then_matmul(
        "GPTQ_gpu_repack_b4_g128",
        "qmatmul_b4_g128",
        "gptq",
    ),
    QuantizationEncoding.Q4_0: _repack_then_matmul(
        "vroom_q4_0_repack_weights",
        "vroom_q4_0_matmul",
        "vroom",
    ),
    QuantizationEncoding.Q4_K: _repack_then_matmul(
        "vroom_q4_k_repack_weights",
        "vroom_q4_k_matmul",
        "vroom",
    ),
    QuantizationEncoding.Q6_K: _repack_then_matmul(
        "vroom_q6_k_repack_weights",
        "vroom_q6_k_matmul",
        "vroom",
    ),
}


def qmatmul(
    encoding: QuantizationEncoding,
    config: Optional[QuantizationConfig],
    lhs: TensorValue,
    *rhs: TensorValue,
) -> TensorValue:
    """Performs matrix multiplication between floating point and quantized
    tensors.

    This quantizes the ``lhs`` floating point value to match the encoding of the
    ``rhs`` quantized value, performs matmul, and then dequantizes the result.
    Beware that, compared to a regular matmul op, this one expects the ``rhs``
    value to be transposed. For example, if the ``lhs`` shape is `[32, 64]`, and
    the quantized ``rhs`` shape is also ``[32, 64]``, then the output shape is
    ``[32, 32]``.

    That is, this function returns the result from:

        dequantize(quantize(lhs) @ transpose(rhs))

    The last two dimensions in ``lhs`` are treated as matrices and multiplied
    by ``rhs`` (which must be a 2D tensor). Any remaining dimensions in ``lhs``
    are broadcast dimensions.

    NOTE: Currently this supports Q4_0, Q4_K, and Q6_K encodings only.

    Args:
        encoding: The quantization encoding to use.
        lhs: The non-quantized, left-hand-side of the matmul.
        *rhs: The transposed and quantized right-hand-side of the matmul and
              auxiliary tensor (if has). Must be rank 2 and in a supported
              [quantization encoding] (/max/api/mojo/graph/quantization/).

    Returns:
        The dequantized result (a floating point tensor).
    """
    if encoding == QuantizationEncoding.GPTQ:
        assert config
        encoding_str = f"{config.quant_method}_b{config.bits}_g{config.group_size}_a{config.desc_act}"
        strategy = _QMATMUL_STRATEGIES.get(encoding_str)
    else:
        strategy = _QMATMUL_STRATEGIES.get(encoding)
    if strategy is None:
        raise ValueError(f"unsupported quantization encoding {encoding}")
    return strategy(lhs, rhs)


_DEQUANTIZE_OP_NAMES: Dict[QuantizationEncoding, str] = {
    QuantizationEncoding.Q4_0: "ggml_q4_0_dequantize",
    QuantizationEncoding.Q4_K: "ggml_q4_k_dequantize",
    QuantizationEncoding.Q6_K: "ggml_q6_k_dequantize",
}


def dequantize(
    encoding: QuantizationEncoding, quantized: TensorValue
) -> TensorValue:
    """Dequantizes a quantized tensor to floating point.

    NOTE: Currently this supports Q4_0, Q4_K, and Q6_K encodings only.

    Args:
        encoding: The quantization encoding to use.
        quantized: The quantized tensor to dequantize.

    Returns:
        The dequantized result (a floating point tensor).
    """
    op_name = _DEQUANTIZE_OP_NAMES.get(encoding)
    if op_name is None:
        raise ValueError(f"unsupported quantization encoding {encoding}")
    *dims, qdim = quantized.shape
    if not isinstance(qdim, StaticDim):
        raise TypeError("dequantize only supported with static last dimension")
    if qdim.dim % encoding.block_size != 0:
        raise ValueError(
            f"last dimension ({qdim}) not divisible by block size "
            f"({encoding.block_size})"
        )
    odim = StaticDim(
        (qdim.dim // encoding.block_size) * encoding.elements_per_block
    )
    flat_quantized = quantized.reshape([-1, qdim])
    flat_dequantized = custom(
        name=op_name,
        values=[flat_quantized],
        out_types=[TensorType(DType.float32, [flat_quantized.shape[0], odim])],
    )[0].tensor
    return flat_dequantized.reshape([*dims, odim])
