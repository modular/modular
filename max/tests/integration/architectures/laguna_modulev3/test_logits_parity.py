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
"""Tests the ModuleV3 Laguna model against the graph architecture, in bfloat16.

See :mod:`laguna_parity` for what the shrunk checkpoint keeps.
"""

from __future__ import annotations

import pytest
from laguna_parity import assert_logits_match


@pytest.mark.parametrize("case_name", ["m1_like"])
def test_logits_match_the_graph_architecture(case_name: str) -> None:
    assert_logits_match(case_name)
