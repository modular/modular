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

"""Optimized quantized operators."""

from max.graph.quantization import (
    Q4_0Encoding,
    Q4_KEncoding,
    Q6_KEncoding,
    QuantizationEncoding,
)

from .custom_ops import custom


@parameter
def _repack_quantized_weights[
    encoding: QuantizationEncoding
](rhs: Symbol) -> Symbol:
    rhs_type = rhs.tensor_type()

    if encoding.id() == Q4_0Encoding.id():
        return ops.custom["vroom_q4_0_repack_weights"](
            List[Symbol](rhs),
            TensorType(DType.uint8, rhs_type.dim(0), rhs_type.dim(1)),
        )
    if encoding.id() == Q4_KEncoding.id():
        return ops.custom["vroom_q4_k_repack_weights"](
            List[Symbol](rhs),
            TensorType(DType.uint8, rhs_type.dim(0), rhs_type.dim(1)),
        )
    if encoding.id() == Q6_KEncoding.id():
        return ops.custom["vroom_q6_k_repack_weights"](
            List[Symbol](rhs),
            TensorType(DType.uint8, rhs_type.dim(0), rhs_type.dim(1)),
        )

    raise "unknown quantization encoding in qmatmul: " + encoding.id()


@parameter
def _packed_qmatmul[
    encoding: QuantizationEncoding
](lhs_matrix: Symbol, rhs_repack: Symbol, rhs_type: TensorType) -> Symbol:
    if encoding.id() == Q4_0Encoding.id():
        return ops.custom["vroom_q4_0_matmul"](
            List[Symbol](lhs_matrix, rhs_repack),
            TensorType(DType.float32, Dim.dynamic(), rhs_type.dim(0)),
        )
    if encoding.id() == Q4_KEncoding.id():
        return ops.custom["vroom_q4_k_matmul"](
            List[Symbol](lhs_matrix, rhs_repack),
            TensorType(DType.float32, Dim.dynamic(), rhs_type.dim(0)),
        )
    if encoding.id() == Q6_KEncoding.id():
        return ops.custom["vroom_q6_k_matmul"](
            List[Symbol](lhs_matrix, rhs_repack),
            TensorType(DType.float32, Dim.dynamic(), rhs_type.dim(0)),
        )

    raise "unknown quantization encoding in qmatmul: " + encoding.id()


def qmatmul[encoding: QuantizationEncoding](lhs: Symbol, rhs: Symbol) -> Symbol:
    """Performs matrix multiplication between floating point and quantized
    tensors.

    This quantizes the `lhs` floating point value to match the encoding of the
    `rhs` quantized value, performs matmul, and then dequantizes the result.
    The operation expects a transposed `rhs` argument, which differs from
    conventional matrix multiplication.

    For matrix shapes:

    - Where standard [`matmul()`](/max/api/mojo/graph/ops/linalg/matmul) expects shapes `($m x $n) @ ($n x $p) → ($m x $p)`
    - `qmatmul()` expects shapes `($m x $n) @ ($p x $n) → ($m x $p)`

    For example, given:

    - lhs shape: [32, 64]
    - rhs shape: [32, 64] (transposed)
    - output shape: [32, 32]

    The operation can be expressed as:

        dequantize(quantize(lhs) . transpose(rhs))

    Where `.` is a normal matmul operator.

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
    rhs_dtype = rhs.type().tensor().dtype
    if rhs_dtype is not DType.uint8:
        raise Error("expected uint8 DType but got ", rhs_dtype)

    g = lhs.graph()
    lhs_type = lhs.tensor_type()
    rhs_type = rhs.tensor_type()
    if rhs_type.rank() != 2:
        raise "rhs must be a matrix"

    # Compute shapes.
    lhs_shape = ops.shape_of(lhs)
    rhs_shape = ops.shape_of(rhs)
    last_lhs_axis = lhs_type.rank() - 1
    reshape_shape = ops.stack(
        List(g.scalar(Int64(-1)), lhs_shape[last_lhs_axis])
    )
    final_shape = ops.concat(List(lhs_shape[:last_lhs_axis], rhs_shape[0:1]))

    # Compute dims for reshape and matmul result types.
    final_dims = List[Dim]()
    for i in range(lhs_type.rank() - 1):
        final_dims.append(lhs_type.dim(i))
    final_dims.append(rhs_type.dim(0))

    matmul_dims = List[Dim]()
    matmul_dims.append(Dim.dynamic())
    matmul_dims.append(lhs_type.dim(-1))

    # Reshape LHS to a matrix, which is expected by the q4_0 matmul op.
    lhs_matrix = ops.reshape(lhs, reshape_shape, matmul_dims)

    # Prepack weights.
    rhs_repack = _repack_quantized_weights[encoding](rhs)

    # Perform quantized matmul.
    qmatmul_out = _packed_qmatmul[encoding](lhs_matrix, rhs_repack, rhs_type)

    # Reshape matmul output to restore the original rank(lhs) - 1 dimensions.
    return ops.reshape(qmatmul_out, final_shape, final_dims)
