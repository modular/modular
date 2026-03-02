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
"""Shared FP8 quantization utilities.

Provides common functions for FP8 scale computation and quantization
used across fused normalization kernels and standalone quantization kernels.
"""

from math import clamp
from utils.numerics import max_finite, min_finite
from sys import is_amd_gpu


@always_inline
fn compute_dynamic_fp8_scale[
    out_dtype: DType,
](
    row_max: Scalar,
    scale_ub: Scalar,
) -> Tuple[
    type_of(scale_ub), type_of(row_max)
]:
    """Compute dynamic FP8 scale factor and its reciprocal from a row max.

    Computes scale_factor = min(row_max, scale_ub) / fp8_max and its reciprocal.
    Does not use `math.recip` to avoid a reciprocal approximation that gives up
    too much precision.

    Parameters:
        out_dtype: The FP8 output dtype (float8_e4m3fn or float8_e4m3fnuz).

    Args:
        row_max: Maximum absolute value across the row/group.
        scale_ub: Upper bound to clamp the scale factor.

    Returns:
        A tuple of (scale_factor, scale_factor_recip).
    """
    comptime assert out_dtype.is_float8(), "out_dtype must be float8"

    comptime fp8_max = max_finite[out_dtype]()
    var scale_factor = (
        min(row_max.cast[scale_ub.dtype](), scale_ub)
        / fp8_max.cast[scale_ub.dtype]()
    )
    var scale_factor_recip = type_of(row_max)(
        0.0 if scale_factor == 0.0 else 1.0 / scale_factor.cast[row_max.dtype]()
    )
    return (scale_factor, scale_factor_recip)


@always_inline
fn fp8_quantize[
    out_dtype: DType,
    *,
    use_clamp: Bool = is_amd_gpu(),
](values: SIMD, scale_recip: Scalar[values.dtype]) -> SIMD[
    out_dtype, values.size
]:
    """Quantize values to FP8, optionally clamping to the representable range.

    On AMD, using clamp is faster because of nan handling.

    Parameters:
        out_dtype: The FP8 output dtype.
        use_clamp: Whether to clamp to [min_finite, max_finite] before cast.
            Defaults to True on AMD GPU, False otherwise.

    Args:
        values: Values to quantize (already normalized as needed, not yet scaled).
        scale_recip: Reciprocal of the FP8 scale factor.

    Returns:
        FP8-quantized values.
    """

    comptime assert out_dtype.is_float8(), "out_dtype must be float8"
    var result = values * scale_recip

    comptime if use_clamp:
        comptime min_val = SIMD[values.dtype, values.size](
            min_finite[out_dtype]()
        )
        comptime max_val = SIMD[values.dtype, values.size](
            max_finite[out_dtype]()
        )
        return clamp(result, min_val, max_val).cast[out_dtype]()
    else:
        return result.cast[out_dtype]()
