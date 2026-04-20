# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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
"""Complex ops."""

from ..dim import StaticDim
from ..value import TensorValue, TensorValueLike
from .reshape import reshape
from .stack import stack


def as_interleaved_complex(x: TensorValueLike) -> TensorValue:
    """Reshapes the input symbolic tensor as complex from alternating (real, imag).

    Args:
        x: A symbolic tensor representing complex numbers as alternating pairs
            of (real, imag) real-valued numbers. Its last dimension must have
            an even size.

    Returns:
        A symbolic tensor representing the complex-valued tensor, but with the
        values pulled out as complex numbers. The result has the same dimensions
        for all dimensions except the last dimension, which is halved,
        and then a final dimension of size 2 representing the complex value.
    """
    g = TensorValue(x)
    shape = g.shape
    last = shape[-1]
    if not isinstance(last, StaticDim):
        raise TypeError("The last dimension must be static.")
    if last.dim % 2 != 0:
        raise ValueError("The last dimension must be divisible by 2.")
    new_shape = shape[:-1] + [last.dim // 2, 2]
    return reshape(g, new_shape)


def mul(lhs: TensorValueLike, rhs: TensorValueLike) -> TensorValue:
    """Multiply two complex valued tensors.

    Complex numbers are represented as a 2-dimensional vector in the
    last dimension.

    Args:
        lhs: A complex number valued symbolic tensor.
        rhs: A complex number valued symbolic tensor.

    Returns:
        The result of multiplying the input values as a complex
        number valued symbolic tensor.
    """
    lhs = TensorValue(lhs)
    rhs = TensorValue(rhs)

    if not (lhs.shape and rhs.shape and lhs.shape[-1] == rhs.shape[-1] == 2):
        raise ValueError(
            "Complex-valued tensors must be 2-dimensional vectors in "
            f"their last dimension. Got {lhs.type=}, {rhs.type=}."
        )

    lhs_real, lhs_imag = lhs[..., 0], lhs[..., 1]
    rhs_real, rhs_imag = rhs[..., 0], rhs[..., 1]
    out_real = (lhs_real * rhs_real) - (lhs_imag * rhs_imag)
    out_imag = (lhs_real * rhs_imag) + (lhs_imag * rhs_real)
    return stack([out_real, out_imag], axis=-1)
