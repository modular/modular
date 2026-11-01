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
"""The compressor's host-side helpers against the reference's formulas.

The pooling itself runs in the registered ``mo.csa.compress`` ops and is
executed on a B200 by the cache-op suite; these tests pin the two helpers
the host runs around it: ``compressed_visibility`` (the reference's
``compress_lens`` mask, one expression) and ``validate_chunk`` (the
odd-start carry contract the op cannot check for itself).
"""

from __future__ import annotations

import numpy as np
import pytest
from max.pipelines.architectures.deepseekV41_modulev3.layers.compressor import (
    compressed_visibility,
    validate_chunk,
)
from max.tensor import Tensor


def test_visibility_is_the_reference_compress_lens() -> None:
    """``j < (q + 1) // ratio`` for both ratios, broadcast over entries."""
    for ratio in (1, 2):
        q = np.arange(16, dtype=np.int32)
        j = np.arange(32, dtype=np.int32)
        got = compressed_visibility(
            Tensor.from_dlpack(q.reshape(-1, 1)),
            Tensor.from_dlpack(j.reshape(1, -1)),
            ratio,
        ).to_numpy()
        expected = (
            np.repeat(j.astype(np.int64)[None, :], 16, axis=0)
            < ((q + 1) // ratio)[:, None]
        )
        np.testing.assert_array_equal(got, expected)


def test_odd_start_validation_requires_a_carry() -> None:
    """An odd-start chunk without its predecessor's row is rejected
    before launch; even starts, carried starts and zero-length requests
    pass."""
    with pytest.raises(ValueError, match="odd"):
        validate_chunk([9], [6], has_carry=[False])
    validate_chunk([8], [6], has_carry=[False])
    validate_chunk([9], [6], has_carry=[True])
    # Zero-length requests never carry, so they pass at any start.
    validate_chunk([9], [0], has_carry=[False])
