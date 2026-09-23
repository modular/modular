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
"""Inkling with MTP PipelineModel: target + chained draft depths in one graph."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from max import tree
from max._core.driver import is_virtual_device_mode
from max.driver import Buffer
from max.graph import Graph, Module
from max.nn.kv_cache import (
    MultiKVCacheParams,
    RecurrentStateInputsPerDevice,
)
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.lib import UnifiedSpecDecodeInputs
from max.pipelines.lib.pipeline_variants.unified_spec_decode_model import (
    _UnifiedSpecDecodeModelMixin,
)
from max.pipelines.speculative import DraftAliases, validate_draft_state_dict
from typing_extensions import override

from ..inkling.batch_processor import InklingInputs
from ..inkling.inkling import kv_collections_by_key
from ..inkling.model import InklingModel
from ..inkling.model_config import (
    STATE_CACHE_KEY,
    InklingConfig,
    nest_inkling_mtp_kv_params,
    parse_inkling_mtp_config,
)
from ..inkling.state_cache import InklingConvScratchPools
from ..inkling.weight_adapters import VISION_PREFIX
from .batch_processor import UnifiedMTPInklingBatchProcessor
from .inkling_mtp import InklingMultiTokenPredictor
from .spec_adapters import (
    DRAFT_CONV_POOLS,
    DRAFT_PRIMARY_KV,
    IMAGE_EMBEDDINGS,
    IMAGE_INDICES,
    POSITIONS,
    TARGET_AUX_KV,
    TARGET_PRIMARY_KV,
    TARGET_STATE,
    split_kv_by_flavor,
)
from .unified_mtp_inkling import UnifiedMTPInkling


@dataclass(kw_only=True)
class UnifiedMTPInklingInputs(UnifiedSpecDecodeInputs, InklingInputs):
    """Inputs for the unified Inkling MTP graph."""

    draft_conv_pools: list[Buffer]

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        assert self.kv_cache_inputs is not None
        canonical = (
            self.tokens,
            self.input_row_offsets,
            self.return_n_logits,
            *self.signal_buffers,
            *tree.leaves(self.kv_cache_inputs),
        )
        # Inkling's own inputs trail the canonical tail; see
        # ``UnifiedMTPInkling.input_types``.
        trailing = (
            self.positions,
            self.image_embeddings,
            self.image_indices,
            *self.draft_conv_pools,
        )
        return (
            canonical
            + self._spec_decode_tail_buffers(include_in_thinking_phase=True)
            + trailing
        )


class UnifiedMTPInklingModel(_UnifiedSpecDecodeModelMixin, InklingModel):
    """Inkling with MTP: merge + target + rejection + chained draft depths."""

    batch_processor_cls: ClassVar[type[UnifiedMTPInklingBatchProcessor]] = (
        UnifiedMTPInklingBatchProcessor
    )

    _draft_state_dict: dict[str, Any]
    _n_mtp_depths: int
    _draft_scratch: InklingConvScratchPools | None
    _fused_nn_model: UnifiedMTPInkling

    def __init__(self, *args, **kwargs):
        kwargs["return_logits"] = ReturnLogits.VARIABLE
        kwargs["return_hidden_states"] = ReturnHiddenStates.ALL_NORMALIZED
        self._draft_scratch = None
        super().__init__(*args, **kwargs)

    @override
    def _load_state_dict(self) -> dict[str, Any]:
        if self.adapter:
            raw_state_dict = self.adapter(
                dict(self.weights.items()),
                huggingface_config=self.huggingface_config,
                pipeline_config=self.pipeline_config,
            )
        else:
            raw_state_dict = {
                key: value.data() for key, value in self.weights.items()
            }

        self._draft_state_dict = {
            k[len("draft.") :]: v
            for k, v in raw_state_dict.items()
            if k.startswith("draft.")
        }
        language = {
            k[len("target.") :]: v
            for k, v in raw_state_dict.items()
            if k.startswith("target.")
        }
        self._vision_weights_dict = {
            name.removeprefix(VISION_PREFIX): data
            for name, data in raw_state_dict.items()
            if name.startswith(VISION_PREFIX)
        }
        self._language_weights_dict = language
        return language

    @override
    def _create_model_config(self, state_dict: dict[str, Any]) -> InklingConfig:
        config = InklingConfig.initialize(
            self.pipeline_config, max_seq_len=self.max_seq_len
        )
        config.finalize(self.huggingface_config, state_dict)
        config.use_subgraphs = False
        mtp = parse_inkling_mtp_config(self.huggingface_config)
        if mtp is None:
            raise ValueError(
                "Inkling MTP requires checkpoint "
                "mtp_config.num_nextn_predict_layers > 0"
            )
        spec = self.pipeline_config.speculative
        assert spec is not None
        self._n_mtp_depths = mtp.num_depths_for(spec)
        # The config's num_speculative_tokens may exceed the checkpoint's
        # depths.
        self.resolved_num_speculative_tokens = self._n_mtp_depths
        config.mtp = mtp

        assert isinstance(self.kv_params, MultiKVCacheParams)
        self.kv_params = nest_inkling_mtp_kv_params(
            self.kv_params, mtp, self._n_mtp_depths
        )
        return config

    @override
    def _wire_batch_processor(
        self, model: Any = None, model_config: Any = None
    ) -> None:
        super()._wire_batch_processor(model, model_config)
        if is_virtual_device_mode():
            return
        self._draft_scratch = InklingConvScratchPools(
            self._fused_nn_model.draft.conv_layout, devices=self.devices
        )
        assert isinstance(
            self._batch_processor, UnifiedMTPInklingBatchProcessor
        )
        self._batch_processor.bind_runtime_state(model, self._draft_scratch)

    @override
    def _build_language_graph(
        self,
        model_config: InklingConfig,
        state_dict: dict[str, Any],
        module: Module,
    ) -> tuple[Graph, dict[str, Any]]:
        del state_dict
        assert self.pipeline_config.speculative is not None
        assert isinstance(self.kv_params, MultiKVCacheParams)
        draft_kv_params = self.kv_params.children["draft"]
        assert isinstance(draft_kv_params, MultiKVCacheParams)
        draft = InklingMultiTokenPredictor(
            model_config, self._n_mtp_depths, draft_kv_params
        )
        nn_model = UnifiedMTPInkling(
            model_config,
            draft,
            speculative_config=self.pipeline_config.speculative,
            enable_structured_output=self.pipeline_config.needs_bitmask_constraints,
        )
        nn_model.draft.embed = nn_model.target.embed
        nn_model.draft.backbone_embed_norm_shards = (
            nn_model.target.embed_norm_shards
        )
        nn_model.target.load_state_dict(
            self._language_weights_dict, weight_alignment=1, strict=True
        )
        nn_model.draft.load_state_dict(
            self._draft_state_dict, weight_alignment=1, strict=False
        )
        self._nn_model = nn_model.target
        self._fused_nn_model = nn_model

        # Inkling names its embedding ``embed``, not ``embed_tokens``.
        validate_draft_state_dict(
            nn_model.draft.raw_state_dict().keys(),
            self._draft_state_dict.keys(),
            DraftAliases(always=("embed.",)),
        )

        weights_registry = {
            **nn_model.draft.state_dict(),
            **nn_model.target.state_dict(),
        }
        kv_params = self.kv_params
        n_devs = len(self.devices)

        with Graph(
            "inkling_with_mtp_graph",
            input_types=nn_model.input_types(kv_params),
            module=module,
        ) as graph:
            graph_inputs = nn_model.decode_inputs(graph.inputs, kv_params)
            kv_tree = graph_inputs.kv_tree
            assert isinstance(kv_tree, Mapping)
            target_tree = kv_tree["target"]
            draft_tree = kv_tree["draft"]
            assert isinstance(target_tree, Mapping)
            assert isinstance(draft_tree, Mapping)
            target_key, target_primary, target_aux = split_kv_by_flavor(
                kv_collections_by_key(target_tree)
            )
            draft_key, draft_primary, draft_aux = split_kv_by_flavor(
                kv_collections_by_key(draft_tree)
            )
            assert STATE_CACHE_KEY in kv_tree, (
                "Inkling always convolves; the cache declared no conv-state"
                " child"
            )
            state = tree.leaves(
                kv_tree[STATE_CACHE_KEY], leaf=RecurrentStateInputsPerDevice
            )

            # Inkling's own inputs, in the order ``input_types`` appends them.
            trailing = iter(graph_inputs.trailing)
            positions = next(trailing).tensor
            image_embeddings = next(trailing).tensor
            image_indices = next(trailing).tensor
            draft_conv_pools = nn_model.draft.conv_layout.take_pools(
                trailing, n_devs
            )

            outputs = nn_model(
                graph_inputs.tokens,
                graph_inputs.input_row_offsets,
                graph_inputs.draft_tokens,
                kv_collections=target_primary,
                draft_kv_collections=draft_primary,
                passthrough_kv=draft_aux,
                return_n_logits=graph_inputs.return_n_logits,
                signal_buffers=graph_inputs.signal_buffers,
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
                extra={
                    POSITIONS: positions,
                    IMAGE_EMBEDDINGS: image_embeddings,
                    IMAGE_INDICES: image_indices,
                    TARGET_STATE: state,
                    DRAFT_CONV_POOLS: draft_conv_pools,
                    TARGET_PRIMARY_KV: target_key,
                    TARGET_AUX_KV: target_aux,
                    DRAFT_PRIMARY_KV: draft_key,
                },
            )
            graph.output(*outputs)

        return graph, weights_registry
