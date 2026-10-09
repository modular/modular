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
from max.serve.scheduler.base import shared_block_ids


def test_returns_the_row_every_leaf_shares() -> None:
    assert shared_block_ids({"full": [3, 4], "full/scales": [3, 4]}) == [3, 4]


def test_empty_mapping_has_no_blocks() -> None:
    assert shared_block_ids({}) == []


def test_rejects_leaves_with_independent_block_ids() -> None:
    # Jenga tiles values and scales separately, so their ids diverge.
    with pytest.raises(ValueError, match="share block ids"):
        shared_block_ids({"full": [1], "full/scales": [20]})
