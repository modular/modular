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

"""Implements the Muse Spark text model using the ModuleV3 API."""

from __future__ import annotations

from typing import Any

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn.common_layers.embedding import VocabParallelEmbedding
from max.experimental.nn.common_layers.linear import ColumnParallelLinear
from max.experimental.nn.common_layers.rotary_embedding import RotaryEmbedding
from max.experimental.nn.sequential import ModuleList
from max.experimental.sharding import DeviceMesh
from max.experimental.tensor import Tensor
from max.nn.legacy.kv_cache import KVCacheParams
from max.pipelines.architectures.gemma4_modulev3.layers.rms_norm import (
    Gemma4RMSNorm,
)
from max.pipelines.architectures.muse_glimmer.muse_glimmer import (
    MuseModelBase,
    MuseTextModelBase,
)

from .layers.attention import NoPERotaryEmbedding
from .layers.transformer_block import MuseSparkTransformerBlock
from .model_config import MuseSparkConfig


def compute_dtype(config: MuseSparkConfig) -> DType:
    """The dtype of everything but the packed NVFP4 experts.

    Under NVFP4 ``config.dtype`` is the packed uint8 of the experts; the
    checkpoint keeps embeddings, norms, attention and the router's inputs in
    bfloat16.
    """
    return DType.bfloat16 if config.text_config.nvfp4 else config.dtype


class MuseSparkTextModel(MuseTextModelBase):
    """The Muse Spark language model (text-only).

    The shared Muse forward (embedding norm, image scatter, softcapped
    logits) with Spark's MoE decoder layers and padded-vocab mask.
    """

    layers: ModuleList[Any]

    def __init__(self, config: MuseSparkConfig, mesh: DeviceMesh) -> None:
        super().__init__()
        text_config = config.text_config
        self.mesh = mesh
        self.dtype = compute_dtype(config)

        self.rope_sliding, self.rope_full = (
            rope_cls(
                dim=text_config.hidden_size,
                n_heads=text_config.num_attention_heads,
                theta=text_config.rope_theta,
                max_seq_len=text_config.max_seq_len,
                device=mesh.devices[0],
                head_dim=text_config.head_dim,
                interleaved=False,
            )
            for rope_cls in (RotaryEmbedding, NoPERotaryEmbedding)
        )

        self.embed_tokens = VocabParallelEmbedding(
            text_config.vocab_size, dim=text_config.hidden_size
        )
        self.embed_norm = Gemma4RMSNorm(
            text_config.hidden_size,
            eps=text_config.rms_norm_eps,
            with_weight=False,
        )
        self.norm = Gemma4RMSNorm(
            text_config.hidden_size, eps=text_config.rms_norm_eps
        )
        self.lm_head = ColumnParallelLinear(
            in_dim=text_config.hidden_size,
            out_dim=text_config.vocab_size,
            bias=False,
        )
        self.output_multiplier = text_config.output_multiplier
        self.logit_softcapping = text_config.final_logit_softcapping
        self.vocab_size = text_config.vocab_size
        self.unpadded_vocab_size = text_config.unpadded_vocab_size

        kv_params_by_type: dict[str, KVCacheParams] = {}
        for layer_type, kv_params_leaf in config.kv_params.children.items():
            assert isinstance(kv_params_leaf, KVCacheParams)
            kv_params_by_type[layer_type] = kv_params_leaf

        layer_type_counts = dict.fromkeys(kv_params_by_type, 0)
        layers = []
        for i, layer_type in enumerate(text_config.layer_types):
            layers.append(
                MuseSparkTransformerBlock(
                    text_config,
                    layer_idx=i,
                    rope=(
                        self.rope_sliding
                        if text_config.use_rope[i]
                        else self.rope_full
                    ),
                    kv_params=kv_params_by_type[layer_type],
                    layer_idx_in_cache=layer_type_counts[layer_type],
                )
            )
            layer_type_counts[layer_type] += 1
        self.layers = ModuleList(layers)
        self._layer_kv_key = list(text_config.layer_types)
        self.return_logits = text_config.return_logits

    def _compute_logits(self, h: Tensor) -> Tensor:
        # lm_head has rows past the tokenizer, and spec decoding samples these
        # logits unbounded, so a padding id would reach detokenization.
        logits = super()._compute_logits(h)
        unpadded = self.unpadded_vocab_size
        if unpadded >= self.vocab_size:
            return logits
        # Placed on the mesh: logits are vocab-sharded under TP.
        keep, drop = (
            F.broadcast_to(
                F.constant(value, DType.float32, device=logits.mesh),
                shape=[size],
            )
            for value, size in (
                (0.0, unpadded),
                (float("-inf"), self.vocab_size - unpadded),
            )
        )
        return logits + F.concat([keep, drop])


class MuseSpark(MuseModelBase):
    """Top-level wrapper: unflattens the two-leaf KV tree, delegates."""

    def __init__(self, config: MuseSparkConfig, mesh: DeviceMesh) -> None:
        super().__init__()
        self.language_model = MuseSparkTextModel(config, mesh)
        self.kv_params = config.kv_params
        self.mesh = mesh
