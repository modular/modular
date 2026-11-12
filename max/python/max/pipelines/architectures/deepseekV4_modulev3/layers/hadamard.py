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
"""Hadamard rotation, the indexer's pre-quantization spread (one matmul)."""

from __future__ import annotations

import functools

import numpy as np
from max import functional as F
from max.dtype import DType
from max.tensor import Tensor

__all__ = ["hadamard_rotate"]


# Extracted from deepseekV4/layers/hadamard.py at dfc8109fc0c.
@functools.lru_cache(maxsize=8)
def _hadamard_matrix(dim: int) -> np.ndarray:
    """Orthonormal Sylvester Hadamard matrix, ``[dim, dim]``.

    Sylvester's construction needs a power of two, which every width this is
    used at satisfies (``index_head_dim`` is 128).
    """
    if dim & (dim - 1):
        raise ValueError(
            f"Hadamard rotation needs a power-of-two width, got {dim}"
        )
    h = np.ones((1, 1), dtype=np.float32)
    while h.shape[0] < dim:
        h = np.block([[h, h], [h, -h]])
    return (h * dim**-0.5).astype(np.float32)


def hadamard_rotate(x: Tensor) -> Tensor:
    """Rotate the last axis of ``x`` by the orthonormal Hadamard matrix."""
    matrix = F.constant(
        _hadamard_matrix(int(x.shape[-1])), DType.float32, device=x.mesh
    )
    return (x.cast(DType.float32) @ matrix).cast(x.dtype)
