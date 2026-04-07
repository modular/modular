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
"""Vision pooling for Gemma4."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from max.dtype import DType
from max.graph import TensorValue, ops
from max.nn.layer import Module


def avg_pool_by_positions(
    all_pixel_position_ids: list[npt.NDArray[np.integer]],
    output_lengths: list[int],
    k: int,
) -> npt.NDArray[np.floating]:
    """Compute sparse average-pooling weights for a ragged batch of images.

    Ports HuggingFace ``Gemma4VisionPooler._avg_pool_by_positions`` to NumPy,
    adapted for ragged (unpadded) multi-image inputs.  Returns a single
    block-diagonal weight matrix covering all images.

    Each real patch at grid position ``(x, y)`` contributes ``1/k**2`` to
    output bin ``(x // k) + (patch_width // k) * (y // k)``.  The
    resulting matrix has one block per image of shape
    ``[output_lengths[i], num_patches_i]``.

    Args:
        all_pixel_position_ids: Per-image list of integer ``(x, y)`` grid
            coordinates, each shaped ``[num_patches_i, 2]``.
        output_lengths: Number of output pooled tokens per image.
        k: Pooling kernel size (``pooling_kernel_size`` from config).

    Returns:
        Weight matrix of shape ``[total_output, total_patches]``, float32,
        where ``total_output = sum(output_lengths)`` and
        ``total_patches = sum(num_patches_i)``.
    """
    patch_counts = [pos.shape[0] for pos in all_pixel_position_ids]
    total_patches = sum(patch_counts)
    total_output = sum(output_lengths)

    weights = np.zeros((total_output, total_patches), dtype=np.float32)

    patch_offset = 0
    row_offset = 0
    inv_k2 = np.float32(1.0 / (k * k))

    for pos_ids, n_patches, n_output in zip(
        all_pixel_position_ids, patch_counts, output_lengths, strict=True
    ):
        max_x = int(pos_ids[:, 0].max()) + 1

        x_bin = pos_ids[:, 0] // k
        y_bin = pos_ids[:, 1] // k
        bin_idxs = x_bin + (max_x // k) * y_bin

        weights[
            row_offset + bin_idxs,
            patch_offset + np.arange(n_patches, dtype=np.intp),
        ] = inv_k2

        patch_offset += n_patches
        row_offset += n_output

    return weights


class Gemma4VisionPooler(Module):
    """Position-based sparse average pooling for the Gemma4 vision encoder.

    Reduces ``total_patches`` patch tokens to ``num_pooled_tokens`` output
    tokens using a pre-computed weight matrix (see :func:`avg_pool_by_positions`)
    passed as a graph input.  There are no learnable parameters.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self._scale: float = math.sqrt(hidden_size)

    def __call__(
        self,
        hidden_states: TensorValue,
        pool_weights: TensorValue,
    ) -> TensorValue:
        """Pool patch tokens using the pre-computed weight matrix.

        Args:
            hidden_states: Packed patch embeddings,
                shape ``[total_patches, hidden_size]``.
            pool_weights: Sparse averaging matrix,
                shape ``[num_pooled_tokens, total_patches]``, dtype bfloat16.

        Returns:
            Pooled embeddings, shape ``[num_pooled_tokens, hidden_size]``.
        """
        original_dtype = hidden_states.dtype
        hidden_states = hidden_states.cast(DType.float32)
        result = (pool_weights @ hidden_states) * ops.constant(
            self._scale, DType.float32, device=hidden_states.device
        )
        return result.cast(original_dtype)
