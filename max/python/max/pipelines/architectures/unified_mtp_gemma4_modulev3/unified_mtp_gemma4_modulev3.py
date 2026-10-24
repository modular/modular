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

"""Gemma4 with MTP speculative decoding, as one ModuleV3 module.

The ModuleV3 counterpart of
``unified_mtp_gemma4.unified_mtp_gemma4.UnifiedMTPGemma4``: the target
(a ``gemma4_modulev3`` :class:`Gemma4TextModel` subclass that appends its
normalized hidden states) and the draft (the ModuleV3 assistant port)
both nest under this module, and a fresh :class:`SequentialDriver` is built
per trace to run the verify/accept/propose loop over them.
"""

from __future__ import annotations

from typing import Any

from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.common_layers.kv_cache import PagedCacheValues
from max.experimental.tensor import Tensor
from max.graph import BufferType, TensorType
from max.nn.kv_cache import KVCacheParamInterface
from max.nn.transformer import ReturnLogits
from max.pipelines.lib.vlm_utils import F_merge_multimodal_embeddings
from max.pipelines.speculative.config import SpeculativeConfig
from max.pipelines.speculative.driver import (
    SequentialDriver,
    gather_variable_rows,
    tensor_compat_verify,
)
from max.pipelines.speculative.spec_input_types import SpecDecodeInputTypeSpec

from ..gemma4.model_config import Gemma4ForConditionalGenerationConfig
from ..gemma4_modulev3.gemma4 import Gemma4TextModel
from .gemma4_assistant_modulev3 import Gemma4AssistantModuleV3
from .spec_adapters import (
    GLOBAL_KV,
    Gemma4ModuleV3MTPProposer,
    Gemma4ModuleV3Target,
)


def gemma4_modulev3_mtp_input_spec(
    config: Gemma4ForConditionalGenerationConfig,
) -> SpecDecodeInputTypeSpec:
    """The signature this arch declares: distributed, with vision inputs.

    Split out of :meth:`UnifiedMTPGemma4ModuleV3.driver` so a test can
    assert the declaration without building the target and draft modules.
    Mirrors the graph-API arch's ``gemma4_mtp_input_spec`` except
    ``trace_owns_signal_buffers=True`` — the ModuleV3 divergence: a
    ``Module.compile``-traced graph appends its own signal buffers after
    every declared input, so the declared signature carries none.
    """
    return SpecDecodeInputTypeSpec(
        devices=config.devices,
        distributed=True,
        data_parallel_degree=1,
        enable_vision=True,
        vision_hidden_size=config.text_config.hidden_size,
        trace_owns_signal_buffers=True,
    )


class UnifiedMTPGemma4ModuleV3Target(Gemma4TextModel):
    """The V3 target, extended to return its normalized hidden states.

    Verification needs every row's post-norm hidden state (the same tensor
    the logits read), so ``forward`` computes the VARIABLE logits out of
    the normalized hidden and appends one per-device tensor of it after
    the standard ``(last_logits, logits, offsets)`` layout.
    """

    def forward(
        self,
        tokens: Tensor,
        sliding_kv: PagedCacheValues,
        global_kv: PagedCacheValues,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
        image_embeddings: Tensor,
        image_token_indices: Tensor,
    ) -> tuple[Tensor, ...]:
        # VARIABLE logits + ALL-normalized hidden states, the contract the
        # spec adapters and driver were built against.
        self.return_logits = ReturnLogits.VARIABLE

        # Stage on every device with one collective each (the inherited
        # broadcast staging: a plain per-device transfer lowers to a
        # host-enqueued stream wait, which is illegal inside device-graph
        # capture).
        tokens = F.distributed_broadcast(tokens, self.mesh)
        input_row_offsets = F.distributed_broadcast(
            input_row_offsets, self.mesh
        )
        image_embeddings = F.distributed_broadcast(image_embeddings, self.mesh)
        image_token_indices = F.distributed_broadcast(
            image_token_indices, self.mesh
        )

        h = self.embed_tokens(tokens)
        self.prepare_freq_cis(self.mesh)

        # Scatter the vision tower's soft tokens over their placeholder
        # positions. Out-of-bounds indices are skipped, so an empty
        # (text-only or decode) batch is a no-op.
        h = F_merge_multimodal_embeddings(
            h, image_embeddings.cast(h.dtype), image_token_indices
        )

        kv_by_type = {
            "sliding_attention": sliding_kv,
            "full_attention": global_kv,
        }
        for idx, layer in enumerate(self.layers):
            h = layer(
                h,
                kv_by_type[self._layer_kv_key[idx]],
                input_row_offsets=input_row_offsets,
            )

        # Every logits read comes from the normalized hidden state, so the
        # norm runs once over the whole merged window.
        norm_h = self.norm(h)

        last_logits = self._compute_logits(
            F.gather(norm_h, input_row_offsets[1:] - 1, axis=0)
        )

        var_h, offsets = gather_variable_rows(
            norm_h, input_row_offsets, return_n_logits, self.mesh
        )
        logits = self._compute_logits(var_h)

        # One normalized hidden-state tensor per device (ALL semantics over
        # the whole merged window), after the standard layout the adapter
        # indexes positionally.
        return (last_logits, logits, offsets, *norm_h.local_shards)


class UnifiedMTPGemma4ModuleV3(Module[..., tuple[Tensor, ...]]):
    """Verifies the previous drafts with the target and proposes the next.

    Holds the target and draft modules plus the driver's construction
    inputs; the driver itself is built per trace (in :meth:`driver`),
    since compiling traces a copy of this module whose parameters are the
    graph's weights and the driver must call that copy's target and draft.
    """

    def __init__(
        self,
        target: UnifiedMTPGemma4ModuleV3Target,
        draft: Gemma4AssistantModuleV3,
        config: Gemma4ForConditionalGenerationConfig,
        *,
        kv_params: KVCacheParamInterface,
        speculative_config: SpeculativeConfig | None,
        enable_structured_output: bool,
        use_greedy_acceptance: bool,
    ) -> None:
        super().__init__()
        self.target = target
        self.draft = draft
        self.config = config
        self.kv_params = kv_params
        self.speculative_config = speculative_config
        self.enable_structured_output = enable_structured_output
        self.use_greedy_acceptance = use_greedy_acceptance

    def driver(self) -> SequentialDriver[Any]:
        """Builds the driver over this module's target and draft.

        Built per trace rather than held: compiling traces a copy of this
        module whose parameters are the graph's weights, and the driver
        must call that copy's target and draft.
        """
        config = self.config
        return SequentialDriver(
            tensor_compat_verify(Gemma4ModuleV3Target(self.target).verify),
            Gemma4ModuleV3MTPProposer(
                self.draft, config.text_config.hidden_size
            ),
            target_model=self.target,
            draft_model=self.draft,
            input_spec=gemma4_modulev3_mtp_input_spec(config),
            speculative_config=self.speculative_config,
            enable_structured_output=self.enable_structured_output,
            use_greedy_acceptance=self.use_greedy_acceptance,
        )

    def input_types(self) -> tuple[TensorType | BufferType, ...]:
        """The types :meth:`forward` takes, one per positional input."""
        return self.driver().input_types(self.kv_params)

    def forward(self, *inputs: Tensor) -> tuple[Tensor, ...]:
        driver = self.driver()
        named = driver.decode_inputs(inputs, self.kv_params)
        # The Gemma4 draft keeps no cache of its own, so the target's
        # sliding collection serves as both the target's and the draft's
        # primary cache; the full-attention leaf rides in passthrough.
        return driver(
            tokens=named.tokens,
            input_row_offsets=named.input_row_offsets,
            vision_embeddings=named.vision_embeddings,
            vision_scatter_indices=named.vision_scatter_indices,
            draft_tokens=named.draft_tokens,
            kv_collections=named.kv("sliding_attention"),
            draft_kv_collections=named.kv("sliding_attention"),
            passthrough_kv={GLOBAL_KV: named.kv("full_attention")},
            return_n_logits=named.return_n_logits,
            host_input_row_offsets=named.host_offsets,
            data_parallel_splits=named.dp_splits,
            batch_context_lengths=named.batch_context_lengths,
            seed=named.seed,
            temperature=named.temperature,
            top_k=named.top_k,
            max_k=named.max_k,
            top_p=named.top_p,
            min_top_p=named.min_top_p,
            in_thinking_phase=named.thinking_phase,
            pinned_bitmask=named.pinned_bitmask,
            wait_payload=named.wait_payload,
            device_bitmask_scratch=named.device_bitmask_scratch,
        )
