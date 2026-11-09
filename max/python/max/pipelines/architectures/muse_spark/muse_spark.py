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

from collections.abc import Iterable
from typing import Any

from max import functional as F
from max.dtype import DType
from max.graph import Value
from max.nn.common_layers.embedding import VocabParallelEmbedding
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.common_layers.linear import ColumnParallelLinear
from max.nn.common_layers.rotary_embedding import RotaryEmbedding
from max.nn.kv_cache import KVCacheParams
from max.nn.legacy.comm.ep import EPBatchManager
from max.nn.sequential import ModuleList
from max.pipelines.architectures.gemma4_modulev3.layers.rms_norm import (
    Gemma4RMSNorm,
)
from max.pipelines.architectures.muse_glimmer.muse_glimmer import (
    MuseModelBase,
    MuseTextModelBase,
)
from max.tensor import Tensor
from max.tensor.sharding import DeviceMapping, DeviceMesh, Sharded

from .layers.attention import NoPERotaryEmbedding
from .layers.moe import expert_quant_config
from .layers.transformer_block import MuseSparkTransformerBlock
from .mesh import bf16_dispatch
from .model_config import MuseSparkConfig


def compute_dtype(config: MuseSparkConfig) -> DType:
    """The dtype of everything but the packed NVFP4 experts.

    Under NVFP4 ``config.dtype`` is the packed uint8 of the experts; the
    checkpoint keeps embeddings, norms, attention and the router's inputs in
    bfloat16.
    """
    return DType.bfloat16 if config.text_config.nvfp4 else config.dtype


def replicate(value: Tensor, mesh: DeviceMesh) -> Tensor:
    """Places a single-device graph input on every device of ``mesh``.

    A per-device ``.to(mesh)`` copy waits on another device's stream, which
    device graph capture rejects (``CUDA_ERROR_STREAM_CAPTURE_MERGE``); the
    broadcast collective does not.
    """
    if value.is_distributed:
        return value
    if mesh.num_devices > 1:
        return F.distributed_broadcast(value, mesh)
    return value.to(mesh)


class MuseSparkTextModel(MuseTextModelBase):
    """The Muse Spark language model (text-only).

    The shared Muse forward (embedding norm, image scatter, softcapped
    logits) with Spark's MoE decoder layers and padded-vocab mask.
    """

    layers: ModuleList[Any]

    def __init__(
        self,
        config: MuseSparkConfig,
        mesh: DeviceMesh,
        ep_batch_manager: EPBatchManager | None = None,
    ) -> None:
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

        self.ep_batch_manager = ep_batch_manager
        self.bf16_ep_batch_manager = (
            EPBatchManager(bf16_dispatch(ep_batch_manager.config))
            if ep_batch_manager is not None
            else None
        )

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
                    ep_batch_manager=(
                        ep_batch_manager
                        if expert_quant_config(text_config, i) is not None
                        else self.bf16_ep_batch_manager
                    ),
                )
            )
            layer_type_counts[layer_type] += 1
        for layer in layers[:-1]:
            layer.defer_join = True
        self.layers = ModuleList(layers)
        self._layer_kv_key = list(text_config.layer_types)
        self.return_logits = text_config.return_logits

    def bind_ep_buffers(self, inputs: Iterable[Tensor | Value[Any]]) -> None:
        """Binds the EP exchange's buffers, which both expert formats share."""
        assert self.ep_batch_manager is not None
        assert self.bf16_ep_batch_manager is not None
        comm = self.ep_batch_manager.comm_buffers(inputs)
        self.bf16_ep_batch_manager.bind_comm_buffers(comm)

    def forward(
        self,
        tokens: Tensor,
        kv_by_type: dict[str, PagedCacheValues],
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
        image_embeddings: Tensor,
        image_token_indices: Tensor,
    ) -> tuple[Tensor, ...]:
        """Runs the language model on mesh-replicated inputs."""
        # Glimmer's own .to(mesh) is then a no-op on the replicated inputs.
        return super().forward(
            replicate(tokens, self.mesh),
            kv_by_type,
            return_n_logits,
            replicate(input_row_offsets, self.mesh),
            replicate(image_embeddings, self.mesh),
            replicate(image_token_indices, self.mesh),
        )

    def _compute_logits(self, h: Tensor) -> Tensor:
        # The pipeline reads one buffer: the vocab-sharded logits gather on
        # the first device.
        return self._mask_padded_tail(super()._compute_logits(h)).to(
            self.mesh.devices[0]
        )

    def _mask_padded_tail(self, logits: Tensor) -> Tensor:
        # lm_head has rows past the tokenizer, and spec decoding samples these
        # logits unbounded, so a padding id would reach detokenization.
        unpadded = self.unpadded_vocab_size
        if unpadded >= self.vocab_size:
            return logits
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
        # Split like the logits' vocabulary, so each device adds its own slice.
        mask = F.concat([keep, drop]).to(
            DeviceMapping(logits.mesh, (Sharded(0),))
        )
        return logits + mask


class MuseSpark(MuseModelBase):
    """Top-level wrapper: unflattens the two-leaf KV tree, delegates.

    Under expert parallelism the EP exchange's buffers follow the KV inputs.
    """

    language_model: MuseSparkTextModel

    def __init__(
        self,
        config: MuseSparkConfig,
        mesh: DeviceMesh,
        ep_batch_manager: EPBatchManager | None = None,
    ) -> None:
        super().__init__()
        self.language_model = MuseSparkTextModel(config, mesh, ep_batch_manager)
        self.kv_params = config.kv_params
        self.mesh = mesh
        self.ep_batch_manager = ep_batch_manager

    def forward(
        self,
        tokens: Tensor,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
        image_embeddings: Tensor,
        image_token_indices: Tensor,
        *variadic_args: Tensor,
    ) -> tuple[Tensor, ...]:
        """Runs the model, binding the trailing EP inputs first."""
        n_ep = (
            len(self.ep_batch_manager.input_types())
            if self.ep_batch_manager is not None
            else 0
        )
        kv_args = variadic_args[: len(variadic_args) - n_ep]
        if self.ep_batch_manager is not None:
            self.language_model.bind_ep_buffers(variadic_args[len(kv_args) :])
        return super().forward(
            tokens,
            return_n_logits,
            input_row_offsets,
            image_embeddings,
            image_token_indices,
            *kv_args,
        )
