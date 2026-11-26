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
"""A decoder's greedy tokens, named beside its logits."""

from __future__ import annotations

from typing import Any

from max.nn.legacy.graph_groups import DecoderOutputs as GroupedOutputs
from max.nn.outputs import ReturnHiddenStates
from max.pipelines.lib.module_registration import (
    _grouped_outputs,
    _role_outputs,
)
from max.pipelines.lib.role_contracts import DecoderOutputs


def test_greedy_tokens_follow_the_logits_in_their_group() -> None:
    logits: Any = object()
    tokens: Any = object()
    grouped = _grouped_outputs(
        DecoderOutputs(next_token_logits=logits, next_tokens=tokens),
        ReturnHiddenStates.NONE,
    )
    assert isinstance(grouped, GroupedOutputs)
    assert list(grouped.logits) == ["next_token_logits", "next_tokens"]
    role = _role_outputs(grouped)
    assert role.next_token_logits is logits
    assert role.next_tokens is tokens


def test_a_decoder_without_greedy_tokens_names_none() -> None:
    logits: Any = object()
    grouped = _grouped_outputs(
        DecoderOutputs(next_token_logits=logits), ReturnHiddenStates.NONE
    )
    assert isinstance(grouped, GroupedOutputs)
    assert list(grouped.logits) == ["next_token_logits"]
    assert _role_outputs(grouped).next_tokens is None
