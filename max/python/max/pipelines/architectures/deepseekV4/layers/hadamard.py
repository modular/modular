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

"""Hadamard rotation, the indexer's pre-quantization spread.

``inference/model.py::rotate_activation``::

    from fast_hadamard_transform import hadamard_transform
    return hadamard_transform(x, scale=x.size(-1) ** -0.5)

``hadamard_transform`` applies the unnormalized Sylvester Walsh-Hadamard matrix
and then multiplies by ``scale``, so with ``scale = d ** -0.5`` the operation is
the orthonormal Hadamard rotation. It is applied before FP4 quantization to
spread outliers across dimensions -- with only eight magnitudes in the e2m1 grid,
a single large component would otherwise dominate its whole block's scale.

MAX has no Hadamard primitive. The transform is a fixed orthogonal matrix, so it
is materialized as a constant and applied with one matmul. That is O(d^2) rather
than the O(d log d) butterfly, which is a performance point and out of scope
(MXSERV-502); at ``d == 128`` the constant is 64 KB.
"""

from __future__ import annotations

import functools

import numpy as np
from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops


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


def hadamard_rotate(x: TensorValue) -> TensorValue:
    """Rotate the last axis of ``x`` by the orthonormal Hadamard matrix."""
    dim = int(x.shape[-1])
    matrix = ops.constant(
        _hadamard_matrix(dim),
        DType.float32,
        x.device if isinstance(x.device, DeviceRef) else DeviceRef.CPU(),
    )
    rotated = ops.matmul(ops.cast(x, DType.float32), matrix)
    return ops.cast(rotated, x.dtype)
