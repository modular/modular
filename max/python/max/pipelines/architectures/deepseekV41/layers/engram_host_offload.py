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
"""Loads Engram tables into pinned host memory that GPUs read in place.

The tables stay in host memory for their whole lifetime; the GPU reads the
rows it needs over PCIe rather than holding a device copy.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from max.driver import Accelerator, Buffer, Usage
from max.dtype import DType
from max.pipelines.weights.fp4_quantization import encode_f32_to_e4m3


def _encode_f32_to_e8m0(
    values: npt.NDArray[np.floating],
) -> npt.NDArray[np.uint8]:
    """Encodes values already on the float8_e8m0fnu grid to their bytes.

    Each value is ``2 ** (byte - 127)``; ``np.frexp`` reads the exponent
    exactly (no ``log2`` rounding).
    """
    _, exponent = np.frexp(np.asarray(values, dtype=np.float64))
    # `frexp` mantissa is in `[0.5, 1)`, so its exponent is one too high.
    return (exponent - 1 + 127).astype(np.uint8)


def load_pinned_engram_table(
    weight: npt.NDArray[np.floating],
    scale: npt.NDArray[np.floating],
    device: Accelerator,
) -> tuple[Buffer, Buffer]:
    """Loads one shard's table rows/scales into pinned host memory.

    Args:
        weight: ``[part_rows, head_dim]`` float values already on the
            ``float8_e4m3fn`` grid.
        scale: ``[part_rows, head_dim // 32]`` float values already on the
            ``float8_e8m0fnu`` grid.
        device: The accelerator the pinned allocation is made against.

    Returns:
        Pinned ``(weight_buf, scale_buf)``, bindable as ``DeviceRef.CPU()``
        graph inputs.
    """
    return pin_engram_table_codes(
        encode_f32_to_e4m3(weight), _encode_f32_to_e8m0(scale), device
    )


def pin_engram_table_codes(
    weight_codes: npt.NDArray[np.uint8],
    scale_codes: npt.NDArray[np.uint8],
    device: Accelerator,
) -> tuple[Buffer, Buffer]:
    """Copies already-encoded table bytes into pinned host memory.

    Byte-level form of :func:`load_pinned_engram_table`; no float32
    intermediate. Only ``device=Accelerator(...)`` with ``Usage.STAGING``
    pins; a ``CPU()`` staging buffer is not pinned. The bytes are written
    through each buffer's ``uint8`` view, since FP8 and E8M0 values cannot
    round-trip through NumPy.

    Args:
        weight_codes: ``[part_rows, head_dim]`` ``float8_e4m3fn`` byte codes.
        scale_codes: ``[part_rows, head_dim // 32]`` ``float8_e8m0fnu`` byte
            codes.
        device: The accelerator the pinned allocation is made against.

    Returns:
        The same pinned ``(weight_buf, scale_buf)`` pair
        :func:`load_pinned_engram_table` returns.
    """
    weight_buf = Buffer(
        shape=weight_codes.shape,
        dtype=DType.float8_e4m3fn,
        device=device,
        usage=Usage.STAGING,
    )
    scale_buf = Buffer(
        shape=scale_codes.shape,
        dtype=DType.float8_e8m0fnu,
        device=device,
        usage=Usage.STAGING,
    )
    weight_buf.view(DType.uint8).inplace_copy_from(
        Buffer.from_numpy(weight_codes)
    )
    scale_buf.view(DType.uint8).inplace_copy_from(
        Buffer.from_numpy(scale_codes)
    )
    return weight_buf, scale_buf
