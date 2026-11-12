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
"""The Laguna target's taps and its variable-logits verify config.

A block drafter reads the residual stream after chosen layers, and a verify
scores the stream through the same head the base graph does. Tapping must
not change the base graph, which the no-spec export compiles.
"""

from __future__ import annotations

import pytest
from laguna_spec_fixtures import build_decoder, stage_decoder, target_config
from max import functional as F
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.outputs import ReturnHiddenStates, ReturnLogits
from max.pipelines.architectures.laguna_modulev3.laguna import LagunaTextModel
from max.pipelines.lib import ModuleV3Outputs
from max.pipelines.lib.interfaces.arch_config import ModelOutputConfig
from max.tensor import Tensor


class _ScoresTheLastTap(LagunaTextModel):
    def forward(
        self,
        tokens: Tensor,
        kv_by_type: dict[str, PagedCacheValues],
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
    ) -> ModuleV3Outputs:
        h, taps = self.run_layers(
            tokens, input_row_offsets, kv_by_type, taps=range(len(self.layers))
        )
        assert len(taps) == len(self.layers)
        assert taps[-1] is h
        last = F.gather(taps[-1], input_row_offsets[1:] - 1, axis=0)
        return ModuleV3Outputs(next_token_logits=self.logits(last))


def test_scoring_the_last_tap_stages_the_base_graph() -> None:
    """Tapping every layer adds no op, and the head over the last tap is
    the base graph's head, so the tapped graph stages the base graph's IR."""
    config = target_config()
    base = stage_decoder(build_decoder(config))
    tapped = build_decoder(config)
    tapped.language_model.__class__ = _ScoresTheLastTap
    assert stage_decoder(tapped) == base


def test_variable_logits_build_only_for_a_verify() -> None:
    target_config(
        ModelOutputConfig(
            return_logits=ReturnLogits.VARIABLE,
            return_hidden_states=ReturnHiddenStates.SELECTED_LAYERS,
            target_layer_ids=(1, 3),
        )
    )
    with pytest.raises(NotImplementedError, match="speculative verify"):
        target_config(ModelOutputConfig(return_logits=ReturnLogits.VARIABLE))


def test_a_tap_past_the_last_layer_is_refused() -> None:
    model = build_decoder(target_config()).language_model
    with F.lazy():
        rows = Tensor.zeros([1])
    with pytest.raises(ValueError, match="taps must index"):
        model.run_layers(rows, rows, {}, taps=[len(model.layers)])
