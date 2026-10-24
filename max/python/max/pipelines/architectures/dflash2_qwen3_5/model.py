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
"""The arch config of the DFlash2 draft architecture.

The DFlash2 draft (HuggingFace architecture ``DFlash2DraftModel``) borrows
its target's ``embed_tokens`` and ``lm_head`` and runs only beside a
target, so it has no pipeline model: its arch config builds the draft's
module for a unified spec-decode graph.
"""

from __future__ import annotations

from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import KVCacheParams
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.lib import KVCacheConfig, PipelineConfig
from max.pipelines.lib.interfaces.pipeline_model import ModelLoadContext
from max.pipelines.lib.utils import parse_state_dict_from_weights
from transformers import AutoConfig

from ..llama3.model_config import Llama3Config
from .dflash2_qwen3_5 import DFlash2Qwen3_5
from .model_config import (
    DRAFT_SLIDING_WINDOW,
    DFlash2Qwen3_5Config,
    parse_dflash2_draft_hf_config,
)


class DFlash2Qwen3_5ArchConfig(Llama3Config):
    """The draft's Llama 3 config, sizing its own windowed KV leaf and
    building the draft's module."""

    @classmethod
    def construct_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
        *,
        allow_kv_head_replication: bool = False,
    ) -> KVCacheParams:
        """Builds the draft's bounded KV leaf from its own geometry.

        Two things differ from the target's leaf and neither can be
        inherited: the drafter's dtype (its K/V projections are bf16 whatever
        the target's cache is) and its window. The window is what makes this
        leaf a per-slot constant: the drafter's mask never reaches further
        back than :data:`DRAFT_SLIDING_WINDOW`, so a page older than the
        window is neither read nor written.
        """
        del cache_dtype, allow_kv_head_replication
        return kv_cache_config.to_params(
            # Pinned rather than following --kv-cache-dtype: the drafter
            # fills this cache with its own unquantized projections. Mach's
            # registry pins the matching group and the MEF sidecar's
            # draft_kv_cache_dtype makes a disagreement a startup error.
            dtype=DType.bfloat16,
            n_kv_heads=huggingface_config.num_key_value_heads,
            head_dim=cls.get_head_dim(huggingface_config),
            num_layers=cls.get_num_layers(huggingface_config),
            devices=devices,
            data_parallel_degree=pipeline_config.model.data_parallel_degree,
            window_size=DRAFT_SLIDING_WINDOW,
        )

    @classmethod
    def build_module(
        cls, context: ModelLoadContext
    ) -> tuple[DFlash2Qwen3_5, DFlash2Qwen3_5Config]:
        """Builds the DFlash2 draft from the draft checkpoint ``context``
        names.

        The draft computes in its checkpoint's own precision and loads
        strictly: a mis-named tensor would load as nothing and show only as
        collapsed acceptance.

        Raises:
            ValueError: If the method isn't ``dflash2``, or the checkpoint's
                tap ids repeat a layer.
        """
        pipeline_config = context.pipeline_config
        speculative = pipeline_config.speculative
        draft_model = pipeline_config.draft_model
        assert speculative is not None and draft_model is not None
        assert draft_model.max_length is not None
        if not speculative.is_dflash2():
            raise ValueError(
                "The DFlash2DraftModel draft is a DFlash2 draft and requires"
                " --speculative-method dflash2, got"
                f" {speculative.speculative_method!r}."
            )
        hf_config = draft_model.huggingface_config
        dflash = parse_dflash2_draft_hf_config(hf_config)
        layers = dflash.target_layer_ids
        if len(set(layers)) != len(layers):
            raise ValueError(
                "DFlash2 target_layer_ids must be distinct: the capture fires"
                " once per tapped layer, so a duplicate yields one fewer"
                f" column than fc expects. Got {layers}."
            )
        state_dict = parse_state_dict_from_weights(
            pipeline_config,
            context.weights,
            context.adapter,
            hf_config=hf_config,
            model_config=draft_model,
        )
        llama = Llama3Config.initialize_from_config(
            pipeline_config,
            hf_config,
            draft_model,
            max_seq_len=draft_model.max_length,
        )
        llama.sliding_window = getattr(hf_config, "sliding_window", None)
        if context.kv_params is not None:
            assert isinstance(context.kv_params, KVCacheParams)
            llama.kv_params = context.kv_params
            llama.devices = list(context.kv_params.devices)
        llama.finalize(
            huggingface_config=hf_config,
            state_dict=state_dict,
            return_logits=ReturnLogits.LAST_TOKEN,
            return_hidden_states=ReturnHiddenStates.LAST,
        )
        draft = DFlash2Qwen3_5(
            llama,
            num_context_features=len(layers),
            block_size=dflash.block_size,
            conv_kernel_size=dflash.conv_kernel_size,
            conv_group_size=dflash.conv_group_size,
            selector_rank=dflash.selector_rank,
            selector_top_k=dflash.selector_top_k,
            layer_types=getattr(hf_config, "layer_types", None) or None,
        )
        draft.load_state_dict(
            state_dict,
            override_quantization_encoding=True,
            weight_alignment=1,
            strict=True,
        )
        return draft, DFlash2Qwen3_5Config(llama, dflash)
