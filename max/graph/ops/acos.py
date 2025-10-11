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
"""Arccosine (inverse cosine) operation."""

import math

from ..value import TensorValue, TensorValueLike
from . import elementwise as ops
from .constant import constant


def acos(x: TensorValueLike) -> TensorValue:
    """Computes the arccosine (inverse cosine) of the input tensor.

    Returns values in the range [0, π] for inputs in [-1, 1].
    Uses polynomial approximation based on the Mojo stdlib implementation,
    employing a Remez approximation with domain splitting to improve accuracy.

    The implementation follows the algorithm from stdlib/math/math.mojo:
    - For F32 and smaller types, uses Remez approximation with domain splitting
    - For F64, delegates to LLVM intrinsic
    - Domain split at |x| = 0.5 for optimal accuracy
    - Special handling for |x| = 1 to avoid numerical instability

    Creates a new op node to compute the elementwise arccosine of a
    symbolic tensor and adds it to the graph, returning the symbolic result.

    .. code-block:: python

        def acos_graph():
            input_type = TensorType(dtype=DType.float32, shape=(3,), device=DeviceRef.CPU())

            with Graph("acos_graph", input_types=(input_type,)) as graph:
                x = graph.inputs[0]
                out = ops.acos(x)
                graph.output(out)

    Args:
        x: Input tensor with values in [-1, 1]. If values are outside this
           domain, they will be clamped to the valid range.

    Returns:
        Arccosine of the input in radians [0, π]. The result will have:
        - the same dtype as the input
        - the same shape as the input

    Raises:
        Error: If the symbol doesn't represent a tensor value.
        Error: If the input is not a floating-point dtype.
    """
    # Clamp input to valid domain [-1, 1]
    x_clamped = ops.max(ops.min(x, 1.0), -1.0)
    x_abs = ops.abs(x_clamped)

    # Create constant tensors with proper dtype and device matching input
    zero = constant(0.0, dtype=x.dtype, device=x.device)
    one = constant(1.0, dtype=x.dtype, device=x.device)
    half = constant(0.5, dtype=x.dtype, device=x.device)
    two = constant(2.0, dtype=x.dtype, device=x.device)
    pi = constant(math.pi, dtype=x.dtype, device=x.device)
    pi_over_2 = constant(math.pi * 0.5, dtype=x.dtype, device=x.device)

    # Domain split at 0.5
    small_domain = x_abs < half

    # Compute x_squared based on domain
    # Small domain: x_squared = x²
    # Large domain: x_squared = (1 - |x|) / 2
    x_squared_small = x_clamped * x_clamped
    x_squared_large = (one - x_abs) * half
    x_squared = ops.where(small_domain, x_squared_small, x_squared_large)

    # Compute d based on domain
    # Small domain: d = |x|
    # Large domain: d = sqrt((1 - |x|) / 2)
    d_small = x_abs
    d_large = ops.sqrt(x_squared_large)
    d = ops.where(small_domain, d_small, d_large)

    # Handle special case |x| = 1 (d should be 0)
    is_one = x_abs >= one
    d = ops.where(is_one, zero, d)

    # Evaluate Remez polynomial using Horner's method
    # Coefficients from Mojo stdlib (Remez approximation)
    c0 = constant(0.4197454825e-1, dtype=x.dtype, device=x.device)
    c1 = constant(0.2424046025e-1, dtype=x.dtype, device=x.device)
    c2 = constant(0.4547423869e-1, dtype=x.dtype, device=x.device)
    c3 = constant(0.7495029271e-1, dtype=x.dtype, device=x.device)
    c4 = constant(0.1666677296e0, dtype=x.dtype, device=x.device)

    poly = c0
    poly = poly * x_squared + c1
    poly = poly * x_squared + c2
    poly = poly * x_squared + c3
    poly = poly * x_squared + c4
    poly = poly * x_squared * d

    # Small domain: π/2 - (d + poly) with sign preservation
    # copysign(d, x) is implemented as d * sign(x)
    is_negative = x_clamped < zero
    sign_x = ops.where(is_negative, -one, one)
    d_signed = d * sign_x
    poly_signed = poly * sign_x
    result_small = pi_over_2 - (d_signed + poly_signed)

    # Large domain: 2 * (d + poly)
    result_large = two * (d + poly)

    # For large domain with negative x: π - result
    result_large = ops.where(is_negative, pi - result_large, result_large)

    # Select based on domain
    result = ops.where(small_domain, result_small, result_large)

    return result
