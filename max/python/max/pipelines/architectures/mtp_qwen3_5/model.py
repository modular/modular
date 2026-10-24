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
"""The MTP head a Qwen3.5 checkpoint ships: its config and module builder.

The head runs only beside its target, inside a unified spec-decode graph.
The config builds the head's module for that graph from the target's own
checkpoint.
"""

from __future__ import annotations

from dataclasses import replace

from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import KVCacheParamInterface, KVCacheParams
from max.pipelines.lib import KVCacheConfig, PipelineConfig
from max.pipelines.lib.interfaces.pipeline_model import (
    ModelLoadContext,
)
from max.pipelines.lib.utils import weights_under_prefix
from transformers import AutoConfig

from ..qwen3_5.model_config import Qwen3_5Config
from ..qwen3_5.mtp import Qwen3_5MTP

__all__ = ["MTPQwen3_5DraftArchConfig"]


class MTPQwen3_5DraftArchConfig(Qwen3_5Config):
    """The head's config: its target's, with a KV leaf for its one layer."""

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
        """Returns a one-layer view of the target's attention cache, for the
        one decoder layer the head has."""
        del allow_kv_head_replication
        attn = Qwen3_5Config._attn_kv_params(
            huggingface_config=huggingface_config,
            pipeline_config=pipeline_config,
            devices=devices,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )
        return replace(attn, num_layers=1)

    @classmethod
    def build_module(
        cls, context: ModelLoadContext
    ) -> tuple[Qwen3_5MTP, Qwen3_5Config]:
        """Builds the MTP head from the target's checkpoint, which
        ``context`` names.

        The head's config is its target's, finalized against the head's
        weights. It embeds and projects through the target's embedding and
        head, so spec decoding shares those onto it.

        Raises:
            ValueError: If the checkpoint ships no MTP head, or one with
                tensors the head doesn't read.
        """
        assert context.kv_params is not None
        # The head's 15 tensors are BF16 in every published checkpoint, the
        # NVFP4 one included, and their names match the module's under the
        # ``mtp.`` prefix. The target's embedding table rides along: the
        # head's config reads the checkpoint's compute dtype off it.
        state_dict = weights_under_prefix(context.weights, _MTP_PREFIX)
        for name in _EMBED_NAMES:
            if context.weights[name].exists():
                state_dict["embed_tokens.weight"] = context.weights[name].data()
        huggingface_config = context.huggingface_config
        config = Qwen3_5Config.initialize_from_config(
            context.pipeline_config,
            huggingface_config,
            max_seq_len=context.max_seq_len,
        )
        assert isinstance(context.kv_params, KVCacheParamInterface)
        config.kv_params = context.kv_params
        config.finalize(
            huggingface_config=Qwen3_5Config._get_text_config(
                huggingface_config
            ),
            state_dict=state_dict,
            return_logits=context.return_logits,
        )
        draft = Qwen3_5MTP(config)
        draft.load_state_dict(state_dict, weight_alignment=1, strict=True)
        return draft, config


_MTP_PREFIX = "mtp."

_EMBED_NAMES = (
    "model.language_model.embed_tokens.weight",
    "model.embed_tokens.weight",
)
"""The target's embedding table, in a multimodal and a text-only checkpoint."""
