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
"""Resolution of the decode context-parallel query layout."""

from __future__ import annotations

import pytest
from max.pipelines.architectures.deepseekV3_2_modulev3.model_config import (
    resolve_dcp_q_replicate,
)
from max.pipelines.lib import MAXModelConfig


def test_unset_replicates_the_query() -> None:
    """No flag means a replicated query, as vLLM defaults GLM-5.x."""
    assert MAXModelConfig.model_fields["dcp_q_replicate"].default is None
    assert resolve_dcp_q_replicate(None) is True


@pytest.mark.parametrize("value", [True, False])
def test_an_explicit_value_wins(value: bool) -> None:
    assert resolve_dcp_q_replicate(value) is value
