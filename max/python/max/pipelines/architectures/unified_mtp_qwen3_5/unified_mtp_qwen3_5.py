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
"""Qwen3.5 with its MTP head, on the shared sequential driver.

The hybrid target makes this graph differ from the other unified MTP graphs in
one structural way: verifying K draft tokens advances 48 Gated DeltaNet
recurrences that no length pointer can rewind. The verify therefore records
each token's update in a ring and folds the accepted prefix into the live pool
-- see :mod:`.state_rollback` and :mod:`.spec_adapters`.

Two smaller Qwen-specific choices, both declared to the driver here:

- Acceptance is **stochastic** and keyed per row, never the greedy fast path,
  because only the stochastic sampler applies the grammar bitmask. This
  checkpoint's ``lm_head`` carries 243 live padding rows, so an unmasked
  argmax can emit an undecodable id -- including at the prefill's position 0,
  which is the one the design flagged.
- The draft's output projection is the target's NVFP4 ``lm_head``, applied at
  one position per request rather than over the whole verified window.
"""

from __future__ import annotations

from max.dtype import DType
from max.graph import BufferType, TensorType, TensorValue
from max.nn.kv_cache import KVCacheParamInterface
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.speculative.config import SpeculativeConfig
from max.pipelines.speculative.driver import SequentialDriver
from max.pipelines.speculative.spec_input_types import SpecDecodeInputTypeSpec
from typing_extensions import override

from ..qwen3_5.model_config import Qwen3_5Config
from ..qwen3_5.mtp import Qwen3_5MTP
from ..qwen3_5.qwen3_5 import Qwen3_5
from ..qwen3_5.state_cache import linear_state_regions, ring_len_for_window
from .spec_adapters import Qwen3_5MTPProposer, Qwen3_5Target
from .spec_state import state_tail_types

__all__ = ["UnifiedMTPQwen3_5", "ring_len_for_config"]


def ring_len_for_config(speculative_config: SpeculativeConfig | None) -> int:
    """Returns the verify ring's record capacity.

    The ring holds one verify window of ``K + 1`` tokens. A step never
    verifies more than the configured draft width, and this architecture
    runs no mixed prefill and decode batch, whose prompt rows would not fit.
    Without a config the driver drafts one token.
    """
    draft_width = (
        1 if speculative_config is None else speculative_config.draft_width
    )
    return ring_len_for_window(1 + draft_width)


class UnifiedMTPQwen3_5(SequentialDriver[list[TensorValue]]):
    """Merge, target verify, state rollback and draft chain, on the driver."""

    target: Qwen3_5
    draft: Qwen3_5MTP

    def __init__(
        self,
        config: Qwen3_5Config,
        speculative_config: SpeculativeConfig | None = None,
        enable_structured_output: bool = False,
    ) -> None:
        if speculative_config is not None:
            if speculative_config.use_greedy_acceptance:
                raise ValueError(
                    "Qwen3.5 MTP requires stochastic acceptance: this"
                    " checkpoint's state rollback and lm_head padding"
                    " exclusion are only validated through the stochastic"
                    " path."
                )
            if speculative_config.synthetic_acceptance_rate is not None:
                raise ValueError(
                    "synthetic acceptance would bypass the state rollback's"
                    " accepted-length plan"
                )

        target = Qwen3_5(config)
        target.return_logits = ReturnLogits.VARIABLE
        target.return_hidden_states = ReturnHiddenStates.ALL_NORMALIZED

        # The draft owns a single-layer KV group, so its layer index is 0. It
        # shares the target's embedding table and rotary cache by reference,
        # which is what makes `mtp_use_dedicated_embeddings: false` structural
        # rather than a load-time convention.
        draft = Qwen3_5MTP(
            config=config,
            embed_tokens=target.embed_tokens,
            rope=target.rope,
            create_norm=target.create_norm,
            kv_layer_idx=0,
        )

        # The same geometry the cache declares, ring included, so the tail
        # matches the pools the engine hands over.
        num_linear_layers = len(target.linear_layer_indices)
        state_regions = linear_state_regions(
            num_linear_layers=num_linear_layers,
            key_head_dim=config.linear_key_head_dim,
            num_key_heads=config.linear_num_key_heads,
            value_head_dim=config.linear_value_head_dim,
            num_value_heads=config.linear_num_value_heads,
            conv_kernel_dim=config.linear_conv_kernel_dim,
            dtype=config.state_dtype,
            num_devices=len(config.devices),
            ring_len=ring_len_for_config(speculative_config),
        )

        target_adapter = Qwen3_5Target(target, state_regions)
        super().__init__(
            target_adapter,
            Qwen3_5MTPProposer(draft, target_adapter, config.hidden_size),
            target_model=target,
            draft_model=draft,
            input_spec=SpecDecodeInputTypeSpec(
                devices=config.devices,
                distributed=True,
                data_parallel_degree=1,
            ),
            speculative_config=speculative_config,
            enable_structured_output=enable_structured_output,
        )
        self.config = config
        self.num_linear_layers = num_linear_layers
        self.state_regions = state_regions

    @override
    @property
    def has_trailing_inputs(self) -> bool:
        # The state-pool tail below.
        return True

    @override
    def input_types(
        self, kv_params: KVCacheParamInterface | None = None
    ) -> tuple[TensorType | BufferType, ...]:
        """Canonical spec-decode signature plus the Qwen state-pool tail.

        The tail is :func:`.spec_state.state_tail_types` over the live leaves
        and the ring, a scratch leaf whose rows the engine supplies like the
        live leaves'. When the target runs M-RoPE a shared
        ``[3, merged_total_seq_len]`` positions tensor follows it.
        """
        config = self.config
        devices = config.devices
        spec_types = super().input_types(kv_params)
        tail = state_tail_types(self.state_regions, devices)

        # M-RoPE positions for the MERGED window -- one column per token of
        # `[real, draft_1..draft_k]` per row, not per prompt token. Shared
        # across devices like the base graph's, and last so the pooled tail
        # above keeps its slot indices.
        #
        # Only declared when the target runs M-RoPE. Without it the rotary
        # falls back to the static table indexed by `cache_length + token_idx`,
        # which is right for text and wrong for every token after an image.
        if self.target.mrope_enabled:
            tail.append(
                TensorType(
                    DType.int64,
                    shape=[3, "merged_total_seq_len"],
                    device=devices[0],
                )
            )

        return (*spec_types, *tail)
