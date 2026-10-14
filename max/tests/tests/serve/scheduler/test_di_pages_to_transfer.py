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

import pytest
from max.serve.scheduler.prefill_scheduler import _pages_to_transfer


def test_each_leaf_keeps_its_own_block_ids() -> None:
    src, dst = _pages_to_transfer(
        {"full": [1, 2], "full/scales": [20, 21]},
        {"full": [7, 8], "full/scales": [52, 53]},
    )

    assert src == {"full": [1, 2], "full/scales": [20, 21]}
    assert dst == {"full": [7, 8], "full/scales": [52, 53]}


def test_skips_pages_decode_holds_in_every_leaf() -> None:
    src, dst = _pages_to_transfer(
        {"full": [1, 2, 3], "full/scales": [20, 21, 22]},
        {"full": [-1, 8, 9], "full/scales": [-1, 53, 54]},
    )

    assert src == {"full": [2, 3], "full/scales": [21, 22]}
    assert dst == {"full": [8, 9], "full/scales": [53, 54]}


def test_rejects_leaves_that_do_not_match() -> None:
    with pytest.raises(ValueError, match="do not match"):
        _pages_to_transfer({"full": [1]}, {"full": [7], "full/scales": [52]})
    with pytest.raises(ValueError, match="pages on prefill"):
        _pages_to_transfer({"full": [1, 2]}, {"full": [7]})
