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
"""Ops that work with complex types.

We don't have a formal complex type yet, so we represent complex numbers
as having a final dimension of size 2, representing the real and complex
parts respectively.
"""

# These operations assume that all tensors have last dim == 2, representing real
# and imaginary parts.

from max.graph.type import Dim
from max.tensor import Tensor, TensorShape

# ===----------------------------------------------------------------------=== #
# Converters
# ===----------------------------------------------------------------------=== #


def as_complex(real: Symbol, imag: Symbol) -> Symbol:
    """Creates a complex-valued tensor from two real-valued tensors.

    Args:
        real: A symbolic tensor representing the real part of the complex value.
        imag: A symbolic tensor representing the imaginary part of the complex value.
            Must have the same shape and dtype as `real`.

    Returns:
        A new symbolic tensor representing the complex valued tensor comprised from
        `real` and `imag`. Each element is paired elementwise.
    """
    # """Builds a complex symbolic tensor from two real-valued symbolic tensors.
    return stack(List[Symbol](real, imag), axis=-1)


def as_interleaved_complex(interleaved: Symbol) -> Symbol:
    """Reshapes the input symbolic tensor as complex from alternating (real, imag).

    Args:
        interleaved: A symbolic tensor representing complex numbers as
            alternating pairs of (real, imag) real-valued numbers. Its last
            dimension must have an even size.

    Returns:
        A symbolic tensor representing the complex-valued tensor, but with the
        values pulled out as complex numbers. The result has the same dimensions
        for all dimensions except the last dimension, which is halved,
        and then a final dimension of size 2 representing the complex value.
    """
    # """Reshape the input tensor as complex, interpreting the last dimension
    # as being alternating (real, imag) pairs."""
    var g = interleaved.graph()
    var interleaved_t = interleaved.tensor_type()
    var last_d = interleaved_t.rank() - 1

    var shape = shape_of(interleaved)
    var back_dims = g.constant(Tensor[DType.int64](TensorShape(2), -1, 2))
    var new_shape = concat(List[Symbol](shape[:last_d], back_dims))

    var new_dims = interleaved_t.dims
    var last_dim = new_dims[last_d]
    new_dims[last_d] = Dim.static(
        last_dim.num_elements() // 2
    ) if last_dim.is_static() else Dim.dynamic()
    new_dims.append(2)

    return reshape(interleaved, new_shape, new_dims)


def as_real(complex: Symbol) -> List[Symbol]:
    """Splits out the real and imaginary components of a symbolic tensor.

    Args:
        complex: The input complex-valued symbolic tensor.

    Returns:
        A pair of real-valued symbolic tensors, each with the same shape and rank
        as the input tensor, except the last dim of size 2 is removed. The first
        represents the real part of the input tensor, and the the second represents
        the imaginary part.
    """
    var splits = split[2](complex, (1, 1), axis=-1)
    var real = splits[0]
    var imag = splits[1]
    return List[Symbol](squeeze(real, axis=-1), squeeze(imag, axis=-1))


# ===----------------------------------------------------------------------=== #
# Ops
# ===----------------------------------------------------------------------=== #


def mul_complex(lhs: Symbol, rhs: Symbol) -> Symbol:
    """Multiplies two complex-valued symbolic tensors elementwise.

    Args:
        lhs: A complex-valued symbolic tensor.
        rhs: A complex-valued symbolic tensor.

    Returns:
        A new complex-valued symbolic tensor. Each element represents
        the elementwize complex multiplication of the element at that location in
        the two input tensors. Type promotion and broadcasting rules are
        applied as described in `elementwise`.
    """
    var lhs_pair = as_real(lhs)
    var rhs_pair = as_real(lhs)
    var l_real = lhs_pair[0]
    var l_imag = lhs_pair[1]
    var r_real = rhs_pair[0]
    var r_imag = rhs_pair[1]

    var out_real = (l_real * r_real) - (l_imag * r_imag)
    var out_imag = (l_real * r_imag) + (l_imag * r_real)
    return as_complex(out_real, out_imag)
