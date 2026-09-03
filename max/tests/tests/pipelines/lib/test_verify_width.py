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
"""The host mirror of the device realize-scatter.

The mirror has to trim the previous step's proposals to this step's verify
width exactly as the device graph does. The previous step always drafted the
configured depth, so once a step verifies fewer than it drafted the two
rectangles stop having the same width -- and mirroring untrimmed raised
"could not broadcast input array from shape (3,) into shape (1,)".
"""

from __future__ import annotations

import numpy as np
from max.pipelines.lib.pipeline_variants.overlap_text_generation import (
    _host_mirror_realized_drafts,
)

_MAGIC = -7


def test_mirror_trims_to_a_narrower_verify_width() -> None:
    """Drafted 3, verifying 1: the tail is dropped, not an error."""
    realized = _host_mirror_realized_drafts(
        np.full((2, 1), _MAGIC, dtype=np.int64),
        np.array([0, 1], dtype=np.int64),
        np.array([[11, 12, 13], [21, 22, 23]], dtype=np.int64),
    )
    np.testing.assert_array_equal(realized, np.array([[11], [21]]))


def test_mirror_leaves_equal_widths_unchanged() -> None:
    prev_next = np.array([[11, 12, 13], [21, 22, 23]], dtype=np.int64)
    realized = _host_mirror_realized_drafts(
        np.full((2, 3), _MAGIC, dtype=np.int64),
        np.array([0, 1], dtype=np.int64),
        prev_next,
    )
    np.testing.assert_array_equal(realized, prev_next)


def test_mirror_of_a_zero_verify_width_is_an_empty_array() -> None:
    """A prefill->decode boundary step verifies nothing."""
    realized = _host_mirror_realized_drafts(
        np.zeros((2, 0), dtype=np.int64),
        np.array([0, 1], dtype=np.int64),
        np.array([[11, 12, 13], [21, 22, 23]], dtype=np.int64),
    )
    assert realized.shape == (2, 0)
