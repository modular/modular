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
"""Qwen3.5-with-MTP PipelineModel: target, draft and state rollback in one graph."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, ClassVar

from max.engine import InferenceSession, Model
from max.graph import Graph, TensorValue
from max.nn.kv_cache import (
    MultiKVCacheParams,
    recurrent_leaf,
)
from max.nn.sampling.penalties import LogitPenalties
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.lib.interfaces.pipeline_model import (
    GraphPipelineModelWithKVCache,
)
from max.pipelines.lib.pipeline_variants.unified_spec_decode_model import (
    _UnifiedSpecDecodeModelMixin,
)
from typing_extensions import override

from ..qwen3_5.model import _SCALE_SUFFIXES, Qwen3_5Model
from ..qwen3_5.model_config import Qwen3_5Config
from ..qwen3_5.state_cache import STATE_CACHE_KEY, attn_cache
from .batch_processor import UnifiedMTPQwen3_5BatchProcessor
from .model_config import UnifiedMTPQwen3_5Config
from .spec_state import POSITION_IDS, graph_kv_params, state_tail
from .unified_mtp_qwen3_5 import UnifiedMTPQwen3_5

logger = logging.getLogger("max.pipelines")

GRAPH_NAME = "qwen3_5_with_mtp_graph"
"""Exported submodel name; the Mach spec-step executor selects it by name."""

_DRAFT_PREFIX = "draft."
_TARGET_PREFIX = "target."


class UnifiedMTPQwen3_5Model(_UnifiedSpecDecodeModelMixin, Qwen3_5Model):
    """Qwen3.5 with MTP: merge, verify, roll the state back, and draft."""

    batch_processor_cls: ClassVar[type[UnifiedMTPQwen3_5BatchProcessor]] = (
        UnifiedMTPQwen3_5BatchProcessor
    )
    # The cache is built from this class, so it must be the one that declares
    # the verify ring.
    model_config_cls: ClassVar[type[Any]] = UnifiedMTPQwen3_5Config

    _draft_state_dict: dict[str, Any]
    _fused_nn_model: UnifiedMTPQwen3_5

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["return_logits"] = ReturnLogits.VARIABLE
        kwargs["return_hidden_states"] = ReturnHiddenStates.ALL_NORMALIZED
        super().__init__(*args, **kwargs)

    @override
    def load_model(self, session: InferenceSession) -> Model:
        """Compiles the one fused graph.

        Skips the base architecture's vision encoder, since this graph is
        text-only.
        """
        return GraphPipelineModelWithKVCache.load_model(self, session)

    @override
    def _wire_batch_processor(
        self, model: Any = None, model_config: Any = None
    ) -> None:
        """Tells the batch processor whether the graph declares positions."""
        super()._wire_batch_processor(model, model_config)
        assert isinstance(
            self._batch_processor, UnifiedMTPQwen3_5BatchProcessor
        )
        self._batch_processor.mrope_enabled = (
            self._fused_nn_model.target.mrope_enabled
        )

    @override
    def _load_state_dict(self) -> dict[str, Any]:
        assert self.adapter is not None, (
            "the unified Qwen3.5 MTP arch requires its safetensors adapter"
        )
        raw = self.adapter(
            dict(self.weights.items()),
            huggingface_config=self.huggingface_config,
            pipeline_config=self.pipeline_config,
        )
        self._draft_state_dict = {
            k[len(_DRAFT_PREFIX) :]: v
            for k, v in raw.items()
            if k.startswith(_DRAFT_PREFIX)
        }
        if not self._draft_state_dict:
            raise ValueError(
                "no mtp.* tensors in the checkpoint; this architecture is only"
                " selected for checkpoints that ship the MTP head"
            )
        return {
            k[len(_TARGET_PREFIX) :]: v
            for k, v in raw.items()
            if k.startswith(_TARGET_PREFIX)
        }

    @override
    def _create_model_config(self, state_dict: dict[str, Any]) -> Qwen3_5Config:
        config = Qwen3_5Config.initialize_from_config(
            self.pipeline_config,
            self.huggingface_config,
            max_seq_len=self.max_seq_len,
        )
        config.finalize(
            huggingface_config=Qwen3_5Config._get_text_config(
                self.huggingface_config
            ),
            state_dict=state_dict,
            return_logits=ReturnLogits.VARIABLE,
            norm_method=self.norm_method,
            attention_bias=self.attention_bias,
        )
        config.tie_word_embeddings = getattr(
            self.huggingface_config, "tie_word_embeddings", False
        )
        # The rollback reads the verify pass's per-layer state-kernel inputs,
        # which cannot cross a subgraph boundary.
        config.use_subgraphs = False
        # No encoder here: it ran before the tokens this graph verifies ever
        # reached it, so compiling one would build a tower nothing calls. The
        # positions are a separate matter -- a request whose context holds an
        # image needs 3-axis positions for every token after it -- so M-RoPE
        # survives the clear.
        config.mrope_without_encoder = config.vision_config is not None
        config.vision_config = None

        # The allocated cache keeps the state child. The graph signature is
        # built from ``graph_kv_params``, which drops it.
        attn = attn_cache(self.kv_params)
        state = recurrent_leaf(self.kv_params)
        assert state is not None, "expected a recurrent state child"
        self.kv_params = MultiKVCacheParams.from_params(
            {
                "target": attn,
                "draft": replace(attn, num_layers=1),
                STATE_CACHE_KEY: state,
            }
        )
        return config

    @override
    def _build_graph_for_compile(
        self,
        session: InferenceSession,
        state_dict: dict[str, Any],
        model_config: Any,
    ) -> tuple[Graph, dict[str, Any]]:
        del session
        assert isinstance(model_config, Qwen3_5Config)
        if not self.pipeline_config.needs_bitmask_constraints:
            raise ValueError(
                "Qwen3.5 MTP needs the constrained-decoding bitmask input:"
                " this checkpoint's lm_head has live padding rows past"
                " sampleable_vocab_size and the in-graph acceptance sampler"
                " excludes them only through that mask. Exporting a MEF with"
                " mach/tools/gen-mef turns it on by default and"
                " --no-sampler-grammar turns it off; elsewhere it follows"
                " --enable-structured-output (or a tool parser that implies"
                " it)."
            )
        nn_model = UnifiedMTPQwen3_5(
            model_config,
            speculative_config=self.pipeline_config.speculative,
            enable_structured_output=self.pipeline_config.needs_bitmask_constraints,
        )

        full_state_dict = _merge_state_dicts(state_dict, self._draft_state_dict)

        _check_weights_match(
            expected=set(nn_model.raw_state_dict().keys()),
            provided=set(full_state_dict.keys()),
        )
        nn_model.load_state_dict(
            full_state_dict,
            override_quantization_encoding=True,
            weight_alignment=1,
            strict=False,
        )
        weights_registry = nn_model.state_dict()
        self.state_dict = weights_registry
        self._fused_nn_model = nn_model

        kv_params = graph_kv_params(self.kv_params)
        num_devices = len(self.devices)

        with Graph(
            GRAPH_NAME, input_types=nn_model.input_types(kv_params)
        ) as graph:
            graph_inputs = nn_model.decode_inputs(graph.inputs, kv_params)
            # Qwen3.5 declares no sparse-attention budget, so
            # batch_context_lengths goes unread.
            trailing = iter(graph_inputs.trailing)

            # The state tail, in the order ``input_types`` declares it.
            state = state_tail(trailing, nn_model.state_regions, num_devices)

            # Declared last by ``input_types`` and only when the target runs
            # M-RoPE, so it is consumed after the whole state tail.
            position_ids: TensorValue | None = None
            if nn_model.target.mrope_enabled:
                position_ids = next(trailing).tensor

            penalties: LogitPenalties | None = None
            if nn_model.logit_penalties:
                penalties = LogitPenalties.from_inputs(
                    [next(trailing).tensor for _ in range(4)]
                )

            outputs = nn_model(
                graph_inputs.tokens,
                graph_inputs.input_row_offsets,
                graph_inputs.draft_tokens,
                kv_collections=graph_inputs.kv("target"),
                draft_kv_collections=graph_inputs.kv("draft"),
                return_n_logits=graph_inputs.return_n_logits,
                signal_buffers=graph_inputs.signal_buffers,
                host_input_row_offsets=graph_inputs.host_offsets,
                data_parallel_splits=graph_inputs.dp_splits,
                seed=graph_inputs.seed,
                temperature=graph_inputs.temperature,
                top_k=graph_inputs.top_k,
                max_k=graph_inputs.max_k,
                top_p=graph_inputs.top_p,
                min_top_p=graph_inputs.min_top_p,
                in_thinking_phase=graph_inputs.thinking_phase,
                pinned_bitmask=graph_inputs.pinned_bitmask,
                wait_payload=graph_inputs.wait_payload,
                device_bitmask_scratch=graph_inputs.device_bitmask_scratch,
                extra={**state, POSITION_IDS: position_ids},
                penalties=penalties,
            )
            graph.output(*outputs)

        return graph, weights_registry


def _merge_state_dicts(
    target: Mapping[str, Any], draft: Mapping[str, Any]
) -> dict[str, Any]:
    """Prefixes both halves into the one flat namespace the graph declares.

    The draft's decoder layer is also ``layers.0.``, so the two halves can
    only be told apart by their module path.

    The draft shares the target's embedding module, and the name walk dedupes
    by module identity, so that weight is declared once, under
    ``target.embed_tokens.weight``. Adding a ``draft.embed_tokens.weight``
    alias here therefore fails the load rather than aliasing anything:
    ``_check_weights_match`` refuses every ``draft.*`` key the graph does not
    consume.

    Args:
        target: Checkpoint tensors for the target, unprefixed.
        draft: Checkpoint tensors for the MTP head, unprefixed.

    Returns:
        The two halves under ``target.`` and ``draft.``. Whether that covers
        what the graph declares depends on the checkpoint, and
        ``_check_weights_match`` is what decides it.
    """
    merged: dict[str, Any] = {
        f"{_TARGET_PREFIX}{name}": value for name, value in target.items()
    }
    merged.update(
        {f"{_DRAFT_PREFIX}{name}": value for name, value in draft.items()}
    )
    return merged


def _check_weights_match(expected: set[str], provided: set[str]) -> None:
    """Fails the load rather than letting ``strict=False`` drop a mismatch.

    Unlike the base architecture's check this grants the MTP head no
    exemption: the fused graph consumes every ``draft.*`` tensor, so an
    unconsumed one means the checkpoint ships a head this graph does not
    implement. Unconsumed target-side tensors keep the base architecture's
    treatment -- a hard failure for quantization scales, whose silent loss
    would leave a quantized layer reading garbage, and a warning for the rest.
    """
    missing = sorted(expected - provided)
    if missing:
        raise ValueError(
            f"Qwen3.5 MTP graph is missing {len(missing)} weight(s): "
            f"{missing[:20]}"
        )

    unused = provided - expected
    unused_draft = sorted(k for k in unused if k.startswith(_DRAFT_PREFIX))
    if unused_draft:
        raise ValueError(
            f"Qwen3.5 MTP checkpoint supplies {len(unused_draft)} MTP-head "
            f"tensor(s) the fused graph does not consume, so this head is not "
            f"the one it implements: {unused_draft[:20]}"
        )
    unused_scales = sorted(k for k in unused if k.endswith(_SCALE_SUFFIXES))
    if unused_scales:
        raise ValueError(
            f"Qwen3.5 MTP checkpoint supplies {len(unused_scales)} "
            f"quantization scale tensor(s) that no layer consumes: "
            f"{unused_scales[:20]}"
        )
    if remaining := sorted(unused - set(unused_scales)):
        logger.warning(
            "Qwen3.5 MTP load_state_dict: %d unused checkpoint keys: %s",
            len(remaining),
            remaining[:20],
        )
