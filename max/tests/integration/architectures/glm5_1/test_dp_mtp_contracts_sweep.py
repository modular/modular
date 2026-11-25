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
"""Checks prefix gathering and decode across the full CPU layout sweep."""

import pytest
from mtp_contract_helpers import (
    check_decode_replica_rows,
    check_verified_prefix,
)
from mtp_contract_helpers import realize_contract as realize_contract


@pytest.mark.parametrize("dp, tp", [(1, 1), (1, 8), (8, 1), (2, 4)])
def test_target_and_draft_gather_each_rows_verified_prefix(
    dp: int, tp: int
) -> None:
    target_hidden_rows = [[[float(rank // tp)]] for rank in range(dp * tp)]
    check_verified_prefix(dp, tp, k=3, target_hidden_rows=target_hidden_rows)


def test_decode_splits_each_replicas_rows_across_its_devices() -> None:
    check_decode_replica_rows(dp=2, tp=4)
