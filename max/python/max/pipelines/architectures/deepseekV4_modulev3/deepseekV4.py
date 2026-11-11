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
"""The DeepSeek-V4-Flash decoder on ModuleV3."""

from __future__ import annotations

from max import functional as F
from max.dtype import DType
from max.graph import TensorValue, ops
from max.nn import Embedding, Linear, Module
from max.nn.kv_cache import KVCacheParamInterface
from max.nn.sequential import ModuleList
from max.pipelines.lib import ModuleV3Outputs
from max.tensor import Tensor, default_dtype
from max.tensor.sharding import DeviceMapping, DeviceMesh

from .layers import (
    DeepseekV4Attention,
    DeepseekV4Cache,
    DeepseekV4MoE,
    DeepseekV4RMSNorm,
    LayerIndex,
    RaggedRows,
    expand_copies,
    hc_head,
    hc_post,
    hyper_connection,
)
from .model_config import DeepseekV4Config

__all__ = ["DeepseekV4", "DeepseekV4Block"]


class DeepseekV4Block(Module[..., Tensor]):
    """One decoder block: mHC-wrapped attention, then mHC-wrapped MoE."""

    def __init__(
        self, config: DeepseekV4Config, layer_idx: int, max_seq_len: int
    ) -> None:
        super().__init__()
        self.attn = DeepseekV4Attention(config, layer_idx, max_seq_len)
        self.ffn = DeepseekV4MoE(config, layer_idx)
        self.attn_norm = DeepseekV4RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.ffn_norm = DeepseekV4RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        # float32 mixer weights: the Sinkhorn chain is not safe in bf16.
        with default_dtype(DType.float32):
            self.hc_attn = hyper_connection(config)
            self.hc_ffn = hyper_connection(config)

    def forward(
        self,
        x: Tensor,
        rows: RaggedRows,
        token_ids: Tensor,
        cache: DeepseekV4Cache,
        index: LayerIndex,
    ) -> Tensor:
        """``[T, hc, d]`` in and out; ``token_ids`` ``[T]`` feed hash routing.

        ``post`` and ``comb`` are read off the stream before each sublayer
        runs, so they describe the incoming stream, not the outgoing one.
        """
        post, comb, h = self.hc_attn(x)
        h = self.attn(self.attn_norm(h), rows, cache, index)
        # The heads' partial sums, completed before the (whole) mHC merge.
        h = h.to(DeviceMapping.replicated(h.mesh))
        x = hc_post(h, x, post, comb)

        post, comb, h = self.hc_ffn(x)
        h = self.ffn(self.ffn_norm(h), token_ids)
        return hc_post(h, x, post, comb)


# Extracted from deepseekV3_modulev3/deepseekV3.py at dfc8109fc0c.
def gather_last_tokens(h: Tensor, input_row_offsets: Tensor) -> Tensor:
    """Gathers the last token of each request from the ragged batch, per shard.

    Under data parallelism ``input_row_offsets`` is rebased into each replica's
    local frame, so its values only address that replica's own rows; the gather
    runs on each device's own rows, never through the auto-sharded
    ``F.gather``.
    """

    def _body(h: TensorValue, offsets: TensorValue) -> TensorValue:
        return ops.gather(h, offsets[1:] - 1, axis=0)

    if not h.is_distributed:
        return F.gather(h, input_row_offsets[1:] - 1, axis=0)
    # The rule-less body runs on each device's own rows; its result keeps h's
    # placements.
    return F.functional(_body)(h, input_row_offsets).rebind_mapping(h.mapping)


class DeepseekV4(Module[..., ModuleV3Outputs]):
    """DeepSeek-V4-Flash with the flattened cache inputs unpacked."""

    def __init__(
        self,
        config: DeepseekV4Config,
        kv_params: KVCacheParamInterface,
        mesh: DeviceMesh,
    ) -> None:
        super().__init__()
        self.config = config
        self.kv_params = kv_params
        self.mesh = mesh
        self.embed = Embedding(config.vocab_size, dim=config.hidden_size)
        self.layers = ModuleList[DeepseekV4Block](
            [
                DeepseekV4Block(config, layer_idx, config.max_seq_len)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = DeepseekV4RMSNorm(config.hidden_size, config.rms_norm_eps)
        # The final mHC contraction; flat names, as the checkpoint has them.
        hc = config.hc_mult
        self.hc_head_fn = Tensor.zeros(
            [hc, hc * config.hidden_size], dtype=DType.float32
        )
        self.hc_head_base = Tensor.zeros([hc], dtype=DType.float32)
        self.hc_head_scale = Tensor.zeros([1], dtype=DType.float32)
        # Stored dtype, not float32: see the V2 ``head_logits``.
        self.head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        tokens: Tensor,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
        *variadic_args: Tensor,
    ) -> ModuleV3Outputs:
        """Runs the trunk on a ragged batch and returns last-token logits."""
        del return_n_logits
        if self.mesh.num_devices > 1:
            tokens = F.distributed_broadcast(tokens, self.mesh)
            input_row_offsets = F.distributed_broadcast(
                input_row_offsets, self.mesh
            )
        else:
            # The broadcast needs signal buffers, which only a multi-device
            # run allocates; onto one device it is a placement change.
            tokens = tokens.to(self.mesh)
            input_row_offsets = input_row_offsets.to(self.mesh)
        cache = DeepseekV4Cache.from_inputs(
            self.config, self.kv_params, variadic_args, tokens.mapping
        )
        rows = RaggedRows.build(
            input_row_offsets,
            cache.cache_lengths,
            tokens.shape[0],
            self.config.window_ratios,
        )

        h = expand_copies(self.embed(tokens), self.config.hc_mult)
        for block in self.layers:
            h = block(h, rows, tokens, cache, block.attn.layer_index())
        h = hc_head(
            h,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.config.rms_norm_eps,
            self.config.hc_eps,
        )
        h = self.norm(gather_last_tokens(h, input_row_offsets))
        logits = h.cast(self.head.weight.dtype) @ self.head.weight.T
        if self.mesh.num_devices > 1:
            # Every device computed them whole (the head is replicated, as in
            # V2); the sampler reads the first device's.
            logits = logits.to(self.mesh.devices[0])
        return ModuleV3Outputs(next_token_logits=logits.cast(DType.float32))
