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
"""The tensor-parallel DeepSeek-V4.1 text trunk."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from max import functional as F
from max import tree
from max.dtype import DType
from max.nn import Module
from max.nn.common_layers.embedding import VocabParallelEmbedding
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.common_layers.linear import ColumnParallelLinear
from max.nn.legacy.kv_cache import KVCacheInputsPerDevice, MultiKVCacheParams
from max.nn.legacy.transformer import ReturnHiddenStates, ReturnLogits
from max.nn.sequential import ModuleList
from max.pipelines.lib import ModuleV3Outputs
from max.tensor import Tensor, default_device, default_dtype
from max.tensor.sharding import DeviceMapping, DeviceMesh

if TYPE_CHECKING:
    from transformers import AutoConfig

from .interface import V41Config, extra_input_names
from .layer_resolver import resolve
from .layers.attention import RaggedRows, absolute_positions, row_ids
from .layers.block import DecoderLayer, leaf_values, slot_constants
from .layers.engram import Engram
from .layers.mhc import mhc_final_collapse, mhc_identity_writeback, mhc_post
from .layers.norm import RMSNorm
from .layers.rope import RopeParams
from .model_config import DeepseekV41Config


class DeepseekV41(Module[..., ModuleV3Outputs]):
    """Runs all live rows through the text decoder, with tensor parallelism."""

    def __init__(
        self,
        config: DeepseekV41Config,
        kv_params: MultiKVCacheParams,
        mesh: DeviceMesh,
        *,
        text_config: AutoConfig,
        token_map: np.ndarray,
    ) -> None:
        super().__init__()
        if config.data_parallel_degree != 1:
            raise ValueError(
                "DeepSeek-V4.1 ModuleV3 currently requires data_parallel_degree=1"
            )
        self.config = config
        self.kv_params = kv_params
        self.mesh = mesh
        self.interface_config = V41Config.from_hf(
            {"text_config": vars(text_config)}
        )
        rope = RopeParams.from_text_config(text_config)
        with default_device(mesh), default_dtype(DType.bfloat16):
            self.embed = VocabParallelEmbedding(
                config.vocab_size, dim=config.hidden_size
            )
            layers = []
            for info in resolve(config):
                engram = None
                if info.layer_id in config.engram_layer_ids:
                    engram = Engram(
                        layer_ids=tuple(config.engram_layer_ids),
                        num_embeddings=tuple(config.engram_num_embeddings),
                        max_ngram_size=int(text_config.engram_max_ngram_size),
                        n_heads=int(text_config.engram_n_heads),
                        head_dim=int(text_config.engram_head_dim),
                        compressed_vocab_size=int(
                            text_config.engram_compressed_vocab_size
                        ),
                        engram_vocab_size=int(text_config.engram_vocab_size),
                        hidden_size=config.hidden_size,
                        hc_mult=config.hc_mult,
                        pad_token_id=int(text_config.engram_pad_token_id),
                        rms_norm_eps=config.rms_norm_eps,
                        device=mesh,
                        token_map=token_map,
                    )
                layers.append(
                    DecoderLayer(
                        config,
                        info,
                        mesh,
                        rope=rope,
                        max_position=config.max_seq_len,
                        candidate_block_size=config.candidate_block_size,
                        engram=engram,
                    )
                )
            self.layers = ModuleList(layers)
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.head = ColumnParallelLinear(
                config.hidden_size, config.vocab_size, bias=False
            )

    def forward(
        self,
        tokens: Tensor,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
        *rest: Tensor,
    ) -> ModuleV3Outputs:
        n_kv = len(self.kv_params.flattened_kv_inputs())
        upstream = self.kv_params.unflatten_kv_inputs(
            iter(x._graph_value for x in rest[:n_kv])
        )
        caches = {
            name: PagedCacheValues.from_upstream(
                tree.leaves(upstream[name], leaf=KVCacheInputsPerDevice),
                DeviceMapping.replicated(self.mesh),
            )
            for name in self.kv_params.children
        }
        names = extra_input_names(self.interface_config)
        if len(rest) != n_kv + len(names):
            raise ValueError(
                "DeepSeek-V4.1 received an unexpected forward input count"
            )
        extras = dict(zip(names, rest[n_kv:], strict=True))
        tokens = F.distributed_broadcast(tokens, self.mesh)
        offsets = F.distributed_broadcast(input_row_offsets, self.mesh)
        positions = absolute_positions(
            offsets, caches["swa"].cache_lengths, tokens.shape[0]
        )
        bid, local = row_ids(offsets, tokens.shape[0])
        rows = RaggedRows(
            offsets,
            bid,
            local,
            positions,
            F.cast(offsets[1:] - offsets[:-1], DType.int32),
        )
        swa_begin = F.distributed_broadcast(extras["swa_begin"], self.mesh)
        floor = F.distributed_broadcast(extras["global_write_floor"], self.mesh)
        dead = F.distributed_broadcast(extras["image_token_mask"], self.mesh)
        lookback = F.distributed_broadcast(extras["engram_lookback"], self.mesh)
        lookback_dead = F.distributed_broadcast(
            extras["engram_lookback_dead"], self.mesh
        )
        tables = [
            extras[f"engram_table_{i}_{part}"]
            for i in range(len(self.config.engram_layer_ids))
            for part in ("weight", "scale")
        ]
        h = self.embed(tokens)
        hc = self.config.hc_mult
        stream = F.stack([h] * hc, axis=1)
        pre = F.concat(
            [
                F.full(
                    [h.shape[0], 1], 1.0, dtype=DType.float32, device=h.mapping
                ),
                F.full(
                    [h.shape[0], hc - 1],
                    0.0,
                    dtype=DType.float32,
                    device=h.mapping,
                ),
            ],
            axis=1,
        )
        y, post, comb = mhc_identity_writeback(stream)
        selection: dict[str, Tensor] = {}
        for layer in self.layers:
            if layer.layer_id in self.config.engram_layer_ids:
                stream = mhc_post(
                    y, stream, post, comb.reshape([stream.shape[0], hc, hc]), hc
                )
                ordinal = F.full(
                    [],
                    self.config.engram_layer_ids.index(layer.layer_id),
                    dtype=DType.uint32,
                    device=stream.mapping,
                )
                tied_offsets = F.rebind(
                    offsets,
                    [lookback.shape[0] + 1],
                    "Engram offsets contain batch+1 entries",
                )
                (stream,) = layer.engram(
                    [],
                    tables,
                    [ordinal, lookback, lookback_dead, dead],
                    hidden=stream,
                    tokens=tokens,
                    positions=positions.cast(DType.int64),
                    query_start_loc=tied_offsets,
                )
                y, post, comb = mhc_identity_writeback(stream)
            routed = {
                "reuse_r2": selection.get("top_k_r2"),
                "reindex_r1": selection.get("pool"),
                "reuse_r1": selection.get("top_k_r1"),
            }.get(layer.group)
            stream, pre, y, post, comb, produced = layer(
                stream,
                pre,
                y,
                post,
                comb,
                rows,
                swa_begin,
                floor,
                dead,
                leaf_values(layer.plan, caches),
                slot_constants(layer.plan),
                [] if routed is None else [routed],
            )
            if layer.group == "full_r2":
                selection["top_k_r2"] = produced[0]
            elif layer.group == "full_r1_cand":
                selection["top_k_r1"], selection["pool"] = produced[:2]
            elif layer.group == "reindex_r1":
                selection["top_k_r1"] = produced[0]
        stream = mhc_post(
            y, stream, post, comb.reshape([stream.shape[0], hc, hc]), hc
        )
        hidden = mhc_final_collapse(stream, pre)
        return self._outputs(hidden, offsets, return_n_logits)

    def _outputs(
        self, hidden: Tensor, offsets: Tensor, return_n_logits: Tensor
    ) -> ModuleV3Outputs:
        last_raw = F.gather(hidden, offsets[1:] - 1, axis=0)
        hidden = self.norm(hidden)
        last = F.gather(hidden, offsets[1:] - 1, axis=0)
        next_logits = self._compute_logits(last)
        logits = None
        logit_offsets = None
        if self.config.return_logits == ReturnLogits.ALL:
            logits = self._compute_logits(hidden)
            logit_offsets = (
                offsets.local_shards[0] if offsets.is_distributed else offsets
            )
        elif self.config.return_logits == ReturnLogits.VARIABLE:
            count = return_n_logits[0]
            suffix = F.range(
                count,
                0,
                -1,
                out_dim="return_n_logits_range",
                device=hidden.mapping,
                dtype=DType.int64,
            )
            indices = (offsets[1:].unsqueeze(-1) - suffix).reshape([-1])
            logits = self._compute_logits(F.gather(hidden, indices, axis=0))
            logit_offsets = F.range(
                0,
                indices.shape[0] + count,
                count,
                out_dim="logit_offsets",
                device=hidden.mapping,
                dtype=DType.int64,
            )
            if logit_offsets.is_distributed:
                logit_offsets = logit_offsets.local_shards[0]
        hidden_states = None
        if self.config.return_hidden_states == ReturnHiddenStates.LAST:
            # Pre-final-norm rows, the same contract as the Llama3
            # ModuleV3 model's ``last_h``; the head consumes the
            # normalized ones.
            hidden_states = (
                last_raw.local_shards[0]
                if last_raw.is_distributed
                else last_raw
            )
        elif (
            self.config.return_hidden_states
            == ReturnHiddenStates.ALL_NORMALIZED
        ):
            hidden_states = (
                hidden.local_shards[0] if hidden.is_distributed else hidden
            )
        return ModuleV3Outputs(
            next_token_logits=next_logits,
            logits=logits,
            logit_offsets=logit_offsets,
            hidden_states=hidden_states,
        )

    def _compute_logits(self, hidden: Tensor) -> Tensor:
        logits = self.head(hidden)
        if logits.is_distributed:
            logits = F.allgather(logits, tensor_axis=-1).local_shards[0]
        return logits.cast(DType.float32)
