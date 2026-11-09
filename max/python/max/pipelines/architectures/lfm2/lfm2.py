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
"""The LFM2 layers and model graph."""

from __future__ import annotations

import functools

from max import tree
from max.dtype import DType
from max.graph import BufferValue, DeviceRef, TensorValue, Weight, ops
from max.nn._graph_ops.state_space import causal_conv1d_varlen_fwd
from max.nn.kv_cache import (
    KVCacheInputsPerDevice,
    PagedCacheValues,
    RecurrentStateInputsPerDevice,
)
from max.nn.legacy.embedding import Embedding
from max.nn.legacy.layer import LayerList, Module
from max.nn.legacy.linear import Linear
from max.nn.legacy.norm import RMSNorm
from max.nn.legacy.rotary_embedding import RotaryEmbedding
from max.nn.legacy.transformer.transformer import logits_postprocess
from max.pipelines.lib.registration_graph_api import decoder_outputs
from max.pipelines.lib.role_contracts import DecoderInputs, DecoderOutputs

from ..llama3.model_config import create_rope_embedding
from .full_attention import LFM2FullAttention
from .model_config import ATTN_CACHE_KEY, STATE_CACHE_KEY, LFM2Config


class LFM2MLP(Module):
    """The gated feed-forward block."""

    def __init__(
        self, config: LFM2Config, linear_cls: functools.partial[Linear]
    ):
        super().__init__()
        self.w1 = linear_cls(
            in_dim=config.hidden_size,
            out_dim=config.intermediate_size,
            dtype=config.dtype,
            device=config.devices[0],
            has_bias=False,
        )
        self.w3 = linear_cls(
            in_dim=config.hidden_size,
            out_dim=config.intermediate_size,
            dtype=config.dtype,
            device=config.devices[0],
            has_bias=False,
        )
        self.w2 = linear_cls(
            in_dim=config.intermediate_size,
            out_dim=config.hidden_size,
            dtype=config.dtype,
            device=config.devices[0],
            has_bias=False,
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        """Runs the gated feed-forward over ``x``."""
        return self.w2(ops.silu(self.w1(x)) * self.w3(x))


class LFM2ShortConv(Module):
    """The short-convolution mixer."""

    def __init__(
        self, config: LFM2Config, linear_cls: functools.partial[Linear]
    ):
        super().__init__()
        hidden = config.hidden_size
        self.hidden_size = hidden
        self.kernel_size = config.conv_L_cache
        self.in_proj = linear_cls(
            in_dim=hidden,
            out_dim=3 * hidden,
            dtype=config.dtype,
            device=config.devices[0],
            has_bias=config.conv_bias,
        )
        self.out_proj = linear_cls(
            in_dim=hidden,
            out_dim=hidden,
            dtype=config.dtype,
            device=config.devices[0],
            has_bias=config.conv_bias,
        )
        self.conv_weight = Weight(
            "conv_weight",
            config.dtype,
            [hidden, 1, self.kernel_size],
            config.devices[0],
        )
        self.conv_bias: Weight | None
        if config.conv_bias:
            self.conv_bias = Weight(
                "conv_bias", config.dtype, [hidden], config.devices[0]
            )
        else:
            self.conv_bias = None

    def __call__(
        self,
        x: TensorValue,
        conv_pool: BufferValue,
        conv_rows: TensorValue,
        has_initial_state: TensorValue,
        input_row_offsets: TensorValue,
    ) -> TensorValue:
        """Runs the gated short conv over a ragged batch.

        The depthwise causal conv (no activation) resumes from each
        request's window of past inputs at ``conv_pool[conv_rows[b]]`` and
        writes the advanced window back in place, so a prefill chunk and a
        decode step run the same op.
        """
        bcx = self.in_proj(x)
        b, c, xv = ops.split(
            bcx,
            [self.hidden_size, self.hidden_size, self.hidden_size],
            axis=-1,
        )
        bx = b * xv
        conv_b = (
            self.conv_bias
            if self.conv_bias is not None
            else ops.constant(0.0, bx.dtype, device=bx.device).broadcast_to(
                [self.hidden_size]
            )
        )
        conv_out = causal_conv1d_varlen_fwd(
            x=bx,
            weight=self.conv_weight.reshape(
                [self.hidden_size, self.kernel_size]
            ),
            bias=conv_b,
            conv_states=conv_pool,
            query_start_loc=ops.cast(input_row_offsets, DType.int32),
            cache_indices=conv_rows,
            has_initial_state=has_initial_state,
            activation="none",
            channels_last=True,
        )
        return self.out_proj(c * conv_out)


class LFM2DecoderLayer(Module):
    """One decoder layer: a mixer, then the feed-forward."""

    def __init__(
        self,
        config: LFM2Config,
        layer_idx: int,
        kv_layer_idx: int,
        rope: RotaryEmbedding,
        linear_cls: functools.partial[Linear],
    ):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        self.kv_layer_idx = kv_layer_idx
        self.operator_norm = RMSNorm(
            config.hidden_size, config.dtype, config.norm_eps
        )
        self.ffn_norm = RMSNorm(
            config.hidden_size, config.dtype, config.norm_eps
        )
        self.feed_forward = LFM2MLP(config, linear_cls)
        self.self_attn: LFM2FullAttention | None
        self.conv: LFM2ShortConv | None
        if self.layer_type == "full_attention":
            self.self_attn = LFM2FullAttention(
                rope=rope,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                hidden_size=config.hidden_size,
                kv_params=config.attn_kv_params,
                layer_idx=kv_layer_idx,
                dtype=config.dtype,
                devices=config.devices,
                linear_cls=linear_cls,
                scale=config.attention_multiplier,
                has_bias=False,
                qk_norm_eps=config.norm_eps,
                norm_dtype=config.dtype,
                attn_output_gate=False,
            )
            self.conv = None
        else:
            self.conv = LFM2ShortConv(config, linear_cls)
            self.self_attn = None

    def __call__(
        self,
        x: TensorValue,
        kv_collection: PagedCacheValues,
        freqs_cis: TensorValue,
        input_row_offsets: TensorValue,
        conv_pool: BufferValue | None = None,
        conv_rows: TensorValue | None = None,
        has_initial_state: TensorValue | None = None,
    ) -> TensorValue:
        """Runs the layer's attention or short convolution, then its MLP."""
        residual = x
        normed = self.operator_norm(x)
        if self.self_attn is not None:
            kv_layer_idx = ops.constant(
                self.kv_layer_idx, DType.uint32, device=DeviceRef.CPU()
            )
            op_out = self.self_attn(
                kv_layer_idx,
                normed,
                kv_collection,
                freqs_cis=freqs_cis,
                input_row_offsets=input_row_offsets,
            )
        else:
            if (
                self.conv is None
                or conv_pool is None
                or conv_rows is None
                or has_initial_state is None
            ):
                raise ValueError(
                    f"Layer {self.layer_type!r} requires conv and its state "
                    "rows to be set, but some are None."
                )
            op_out = self.conv(
                normed,
                conv_pool,
                conv_rows,
                has_initial_state,
                input_row_offsets,
            )

        h = residual + op_out
        h = h + self.feed_forward(self.ffn_norm(h))
        return h


class LFM2(Module):
    """The LFM2 model, registered for the ``DECODER`` role.

    Its convolution layers keep each request's window of past inputs in the
    KV cache's state leaf, beside the attention layers' KV.
    """

    graph_name = "lfm2"
    """The name the graph compiled under before registration, which keys
    precompiled MEFs."""

    override_quantization_encoding = True

    def __init__(self, config: LFM2Config):
        super().__init__()
        assert len(config.devices) == 1
        self.config = config
        rope = create_rope_embedding(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            rope_theta=config.rope_theta,
            max_seq_len=config.max_seq_len,
            interleaved_rope_weights=False,
            rope_scaling_params=None,
            longrope_scaling_params=None,
            device=config.devices[0],
        )
        linear_cls = functools.partial(Linear, quant_config=config.quant_config)
        self.rope = rope
        kv_idx = 0
        layers: list[LFM2DecoderLayer] = []
        for i, t in enumerate(config.layer_types):
            layer = LFM2DecoderLayer(
                config=config,
                layer_idx=i,
                kv_layer_idx=kv_idx,
                rope=rope,
                linear_cls=linear_cls,
            )
            if t == "full_attention":
                kv_idx += 1
            layers.append(layer)
        self.layers = LayerList(layers)
        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            config.dtype,
            config.devices[0],
        )
        self.norm = RMSNorm(config.hidden_size, config.dtype, config.norm_eps)
        self.lm_head = Linear(
            config.hidden_size,
            config.vocab_size,
            config.dtype,
            config.devices[0],
            has_bias=False,
        )
        if config.tie_word_embeddings:
            self.lm_head.set_shared_weight("weight", self.embed_tokens.weight)

    def __call__(
        self, inputs: DecoderInputs[TensorValue]
    ) -> DecoderOutputs[TensorValue]:
        """Runs the model over the batch and returns the logits.

        Each convolution layer advances its rows of the state leaf in place.
        """
        kv_tree = self.config.kv_params.unflatten_kv_inputs(
            iter(tree.leaves(inputs.kv))
        )
        (kv_collection,) = tree.leaves(
            kv_tree[ATTN_CACHE_KEY], leaf=KVCacheInputsPerDevice
        )
        (state,) = tree.leaves(
            kv_tree[STATE_CACHE_KEY], leaf=RecurrentStateInputsPerDevice
        )
        (window,) = state.leaves
        input_row_offsets = inputs.input_row_offsets
        h = self.embed_tokens(inputs.tokens)
        freqs_cis = self.rope.freqs_cis.to(h.device)
        # The KV cache manager zeroes a request's state rows before its first
        # forward, so every request can resume from its rows.
        has_initial_state = ops.broadcast_to(
            ops.constant(True, DType.bool, device=h.device),
            [window.live_row_ids.shape[1]],
        )
        conv_idx = 0
        for layer in self.layers:
            if layer.layer_type == "full_attention":
                h = layer(
                    h,
                    kv_collection=kv_collection,
                    freqs_cis=freqs_cis,
                    input_row_offsets=input_row_offsets,
                )
                continue
            h = layer(
                h,
                kv_collection=kv_collection,
                freqs_cis=freqs_cis,
                input_row_offsets=input_row_offsets,
                conv_pool=window.pool,
                conv_rows=window.live_row_id(conv_idx),
                has_initial_state=has_initial_state,
            )
            conv_idx += 1

        outputs = logits_postprocess(
            h=h,
            input_row_offsets=input_row_offsets,
            return_n_logits=inputs.return_n_logits,
            norm=self.norm,
            lm_head=self.lm_head,
            return_logits=self.config.return_logits,
            return_hidden_states=self.config.return_hidden_states,
            logits_scaling=self.config.logits_scaling,
        )
        return decoder_outputs(outputs)
