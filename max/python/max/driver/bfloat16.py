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
"""NumPy interop for bfloat16, which NumPy can't represent natively.

NumPy has no bfloat16 dtype, so these functions carry bfloat16 values as their
bit patterns in a ``uint16`` array. Pair them with :meth:`Buffer.view` to move
those bits in and out of a bfloat16 :class:`Buffer`.
"""

import numpy as np
import numpy.typing as npt
from max.dtype import DType

from .buffer import Buffer
from .driver import Device


def bfloat16_as_uint16_to_float32(
    arr: npt.NDArray[np.uint16],
) -> npt.NDArray[np.float32]:
    """Converts bfloat16 bit patterns (stored as ``uint16``) to ``float32``.

    The conversion is exact: a bfloat16 value is the top 16 bits of a
    ``float32``.

    For example, read a bfloat16 buffer back as ``float32``:

    .. code-block:: python

        import numpy as np
        from max.driver import (
            Buffer,
            bfloat16_as_uint16_to_float32,
            float32_to_bfloat16_as_uint16,
        )
        from max.dtype import DType

        bits = float32_to_bfloat16_as_uint16(np.array([1.5], np.float32))
        buffer = Buffer.from_numpy(bits).view(DType.bfloat16)
        values = bfloat16_as_uint16_to_float32(
            buffer.view(DType.uint16).to_numpy()
        )

    Args:
        arr: A ``uint16`` array holding bfloat16 bit patterns.

    Returns:
        A ``float32`` array with the same shape as ``arr``.
    """
    assert arr.dtype == np.uint16, f"Expected uint16, got {arr.dtype}"
    widened = arr.astype(np.uint32)
    widened <<= 16
    return widened.view(np.float32)


def float32_to_bfloat16_as_uint16(
    arr: npt.NDArray[np.float32],
) -> npt.NDArray[np.uint16]:
    """Converts a ``float32`` array to bfloat16 bit patterns stored as ``uint16``.

    Rounds to nearest, ties to even, the same as a compiled ``float32`` to
    bfloat16 cast. A NaN input becomes the canonical quiet NaN with the input's
    sign.

    Args:
        arr: A ``float32`` array.

    Returns:
        A ``uint16`` array of bfloat16 bit patterns with the same shape as
        ``arr``.
    """
    assert arr.dtype == np.float32, f"Expected float32, got {arr.dtype}"

    flat = np.ascontiguousarray(arr).reshape(-1)
    bits = flat.view(np.uint32)

    # Round to nearest even: add (0x7FFF + bit16) then shift right 16.
    # bit16 is the LSB of the upper 16 bits, used for round-to-even on ties.
    # This is equivalent to: round_up when lower > 0x8000, or when
    # lower == 0x8000 and bit16 == 1 (round ties to even).
    rounded = bits >> 16
    rounded &= 1
    rounded += 0x7FFF
    rounded += bits
    rounded >>= 16
    narrowed = rounded.astype(np.uint16)

    # A NaN's exponent is already saturated, so the rounding carry escapes the
    # mantissa and lands in it, silently producing a finite value: 0x7F800001
    # rounded to +inf and 0x7FFFFFFF to -0.0. Emit the canonical quiet NaN
    # instead, keeping only the sign.
    is_nan = np.isnan(flat)
    if is_nan.any():
        narrowed[is_nan] = ((bits[is_nan] >> 16) & 0x8000) | 0x7FC0

    return narrowed.reshape(arr.shape)


def float32_array_to_buffer(
    array: npt.NDArray[np.float32],
    *,
    dtype: DType,
    device: Device,
) -> Buffer:
    """Creates a :class:`Buffer` of ``dtype`` on ``device`` from ``float32`` data.

    Converts on the host, so no cast graph is compiled. For
    :obj:`DType.bfloat16`, rounds with :func:`float32_to_bfloat16_as_uint16`;
    for other dtypes, casts with NumPy.

    Args:
        array: The ``float32`` host data.
        dtype: The element type of the returned buffer.
        device: The device to place the returned buffer on.

    Returns:
        A buffer with the same shape as ``array``.
    """
    array = np.ascontiguousarray(array, dtype=np.float32)

    if dtype == DType.bfloat16:
        return (
            Buffer.from_numpy(float32_to_bfloat16_as_uint16(array))
            .view(DType.bfloat16, array.shape)
            .to(device)
        )

    if dtype == DType.float32:
        return Buffer.from_numpy(array).to(device)

    return Buffer.from_numpy(array.astype(dtype.to_numpy(), copy=False)).to(
        device
    )
