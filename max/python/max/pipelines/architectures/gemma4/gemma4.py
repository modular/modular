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

"""Implements the Gemma4 model."""

from __future__ import annotations

import functools
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from max import tree
from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    Graph,
    ShardingStrategy,
    TensorValue,
    Type,
    Value,
    ops,
)
from max.graph import Module as GraphModule
from max.nn.graph_groups import HiddenStates
from max.nn.kv_cache import (
    KVCacheParamInterface,
    KVCacheParams,
    MultiKVCacheParams,
    PagedCacheValues,
)
from max.nn.layer import LayerList, MissingLayer, Module
from max.nn.linear import MLP, ColumnParallelLinear
from max.nn.moe import GatedActivation, MoE, MoEQuantized
from max.nn.rotary_embedding import Llama3RotaryEmbedding
from max.nn.transformer import DecoderOutputs, ReturnHiddenStates
from max.nn.transformer.distributed_transformer import (
    DistributedLogitsPostprocessMixin,
)
from max.pipelines.architectures.gemma3.layers.scaled_word_embedding import (
    ScaledWordEmbedding,
)
from max.pipelines.lib.interfaces.batch_processor import (
    ragged_batch_inputs,
    signal_inputs,
    vision_merge_inputs,
)
from max.pipelines.lib.vlm_utils import merge_multimodal_embeddings

from .layers.attention import Gemma4Attention
from .layers.decoder_layer import Gemma4TextDecoderLayer
from .layers.moe import Gemma4MoE, Gemma4MoEGate
from .layers.rms_norm import Gemma4RMSNorm
from .layers.rotary_embedding import ProportionalRotaryEmbedding
from .model_config import Gemma4ForConditionalGenerationConfig


def language_input_tree(
    devices: Sequence[DeviceRef],
    dtype: DType,
    hidden_size: int,
    kv_params: KVCacheParamInterface,
) -> OrderedDict[str, object]:
    """Names the language graph's inputs; its leaves are its inputs.

    Image embeddings are always inputs: a text-only batch feeds zero rows.

    Args:
        devices: The devices the graph runs on.
        dtype: The dtype of the image embeddings.
        hidden_size: The width of each image embedding row.
        kv_params: The KV cache parameters.

    Returns:
        The input types under ``batch``, ``vision``, ``signals`` and ``kv``,
        in graph order.
    """
    return OrderedDict(
        batch=ragged_batch_inputs(
            devices[0], ("tokens", "return_n_logits", "row_offsets")
        ),
        vision=vision_merge_inputs(
            devices, dtype, hidden_size, rows_dim="num_image_tokens"
        ),
        signals=signal_inputs(list(devices)),
        kv=kv_params.get_symbolic_inputs(),
    )


class Gemma4TextModel(DistributedLogitsPostprocessMixin, Module):
    """The Gemma 4 language model."""

    def __init__(self, config: Gemma4ForConditionalGenerationConfig) -> None:
        super().__init__()
        text_config = config.text_config
        self.devices = config.devices
        self._embedding_dtype = config.unquantized_dtype
        # The decoder layers this instance builds, all of them unless it is a
        # pipeline stage. self.layers holds a MissingLayer at every other
        # position, so weight names and layer indices stay global.
        self.layer_ids = config.layer_ids or range(
            text_config.num_hidden_layers
        )
        self.holds_first = self.layer_ids.start == 0
        self.holds_last = self.layer_ids.stop == text_config.num_hidden_layers

        # Build per-layer-type rotary embeddings.
        # sliding_attention uses default rope, full_attention uses proportional.
        rope_sliding = Llama3RotaryEmbedding(
            dim=text_config.hidden_size,
            n_heads=text_config.num_attention_heads,
            theta=text_config.sliding_window_rope_theta,
            max_seq_len=text_config.max_position_embeddings,
            head_dim=text_config.head_dim,
            interleaved=False,
            scaling_params=None,
        )

        rope_global = ProportionalRotaryEmbedding(
            dim=text_config.hidden_size,
            n_heads=text_config.num_attention_heads,
            theta=text_config.global_rope_theta,
            max_seq_len=text_config.max_position_embeddings,
            head_dim=text_config.global_head_dim,
            interleaved=False,
            scaling_params=text_config.global_rope_scaling,
        )

        unquantized_dtype = config.unquantized_dtype

        embedding_output_dtype = config.dtype
        quant_config = text_config.quant_config
        if quant_config and quant_config.embedding_output_dtype:
            embedding_output_dtype = quant_config.embedding_output_dtype

        if self.holds_first or (self.holds_last and config.tie_word_embeddings):
            self.embed_tokens = ScaledWordEmbedding(
                text_config.vocab_size,
                text_config.hidden_size,
                embedding_output_dtype,
                config.devices,
                embed_scale=text_config.hidden_size**0.5,
            )

        if self.holds_last:
            self.norm = Gemma4RMSNorm(
                text_config.hidden_size,
                unquantized_dtype,
                text_config.rms_norm_eps,
            )
            self.norm.sharding_strategy = ShardingStrategy.replicate(
                len(config.devices)
            )
            self.norm_shards = self.norm.shard(config.devices)

            self.lm_head = ColumnParallelLinear(
                text_config.hidden_size,
                text_config.vocab_size,
                dtype=unquantized_dtype,
                devices=config.devices,
                tied_weight=(
                    self.embed_tokens.weight
                    if config.tie_word_embeddings
                    else None
                ),
            )

        # Resolve per-layer KVCacheParams from MultiKVCacheParams. The tree is
        # keyed by layer-type name ("sliding_attention" / "full_attention").
        assert isinstance(config.kv_params, MultiKVCacheParams)
        kv_params_by_layer_type: dict[str, KVCacheParams] = {}
        for _k, _p in config.kv_params.children.items():
            assert isinstance(_p, KVCacheParams)
            kv_params_by_layer_type[_k] = _p

        layer_type_counts: dict[str, int] = {
            "sliding_attention": 0,
            "full_attention": 0,
        }
        layers: dict[int, Gemma4TextDecoderLayer] = {}
        for i in self.layer_ids:
            layer_type = text_config.layer_types[i]
            kv_params = kv_params_by_layer_type[layer_type]

            layer_idx_in_cache = layer_type_counts[layer_type]
            layer_type_counts[layer_type] += 1
            is_sliding = layer_type == "sliding_attention"

            is_nvfp4 = quant_config is not None and quant_config.is_nvfp4
            moe_nvfp4 = is_nvfp4 and text_config.enable_moe_block
            # The first-party NVFP4 checkpoints (nvidia/Gemma-4-*) keep
            # attention in BF16, but other modelopt quants (e.g. the
            # community 12B NVFP4 ones) quantize it too -- honor the
            # per-layer classification instead of assuming BF16.
            # is_nvfp4 already implies quant_config is not None.
            attn_quantized = (
                is_nvfp4
                and quant_config is not None
                and i in quant_config.attn_quantized_layers
            )

            moe_block: MoE | None = None
            if text_config.enable_moe_block:
                moe_gate_cls = functools.partial(
                    Gemma4MoEGate,
                    eps=text_config.rms_norm_eps,
                )
                moe_norm_cls = functools.partial(
                    Gemma4RMSNorm,
                    text_config.hidden_size,
                    unquantized_dtype,
                    eps=text_config.rms_norm_eps,
                )
                if is_nvfp4:
                    moe_block = MoEQuantized(
                        devices=config.devices,
                        hidden_dim=text_config.hidden_size,
                        num_experts=text_config.num_experts,
                        num_experts_per_token=text_config.top_k_experts,
                        moe_dim=text_config.moe_intermediate_size,
                        gate_cls=moe_gate_cls,
                        activation=GatedActivation("gelu_tanh"),
                        pre_expert_norm_cls=moe_norm_cls,
                        dtype=config.dtype,
                        quant_config=quant_config,
                    )
                else:
                    moe_block = Gemma4MoE(
                        devices=config.devices,
                        hidden_dim=text_config.hidden_size,
                        num_experts=text_config.num_experts,
                        num_experts_per_token=text_config.top_k_experts,
                        moe_dim=text_config.moe_intermediate_size,
                        gate_cls=moe_gate_cls,
                        activation=GatedActivation("gelu_tanh"),
                        pre_expert_norm_cls=moe_norm_cls,
                        dtype=config.dtype,
                    )

            layers[i] = Gemma4TextDecoderLayer(
                attention=Gemma4Attention(
                    rope_global=rope_global,
                    rope_local=rope_sliding,
                    num_attention_heads=text_config.num_attention_heads,
                    num_key_value_heads=text_config.num_key_value_heads,
                    num_global_key_value_heads=text_config.num_global_key_value_heads,
                    attention_k_eq_v=text_config.attention_k_eq_v,
                    hidden_size=text_config.hidden_size,
                    kv_params=kv_params,
                    global_head_dim=text_config.global_head_dim,
                    layer_idx=i,
                    layer_idx_in_cache=layer_idx_in_cache,
                    is_sliding=is_sliding,
                    dtype=unquantized_dtype
                    if (is_nvfp4 and not attn_quantized)
                    else config.dtype,
                    devices=config.devices,
                    qk_norm_eps=text_config.rms_norm_eps,
                    local_window_size=text_config.sliding_window,
                    quant_config=None
                    if (is_nvfp4 and not attn_quantized)
                    else quant_config,
                ),
                mlp=MLP(
                    dtype=unquantized_dtype if moe_nvfp4 else config.dtype,
                    quantization_encoding=None,
                    hidden_dim=text_config.hidden_size,
                    feed_forward_length=text_config.intermediate_size,
                    devices=config.devices,
                    activation_function=text_config.hidden_activation,
                    quant_config=None if moe_nvfp4 else quant_config,
                ),
                hidden_size=text_config.hidden_size,
                rms_norm_eps=text_config.rms_norm_eps,
                devices=config.devices,
                unquantized_dtype=unquantized_dtype,
                enable_moe_block=text_config.enable_moe_block,
                moe_block=moe_block,
            )

        # Store the per-layer cache-tree key ("sliding_attention" /
        # "full_attention") so __call__ can route the correct cache to each
        # layer.
        self._layer_kv_key = [
            text_config.layer_types[i]
            for i in range(text_config.num_hidden_layers)
        ]

        self.dim = text_config.hidden_size
        self.n_heads = text_config.num_attention_heads
        self.layers = LayerList(
            [
                layers[i] if i in self.layer_ids else MissingLayer(i)
                for i in range(text_config.num_hidden_layers)
            ]
        )
        self.kv_params = config.kv_params
        self.return_logits = text_config.return_logits
        self.return_hidden_states = text_config.return_hidden_states
        self.target_layer_ids: list[int] | None = (
            list(text_config.target_layer_ids)
            if text_config.target_layer_ids
            else None
        )
        if self.target_layer_ids is not None:
            n_layers = text_config.num_hidden_layers
            for pos, layer_id in enumerate(self.target_layer_ids):
                if not 0 <= layer_id < n_layers:
                    raise ValueError(
                        f"target_layer_ids[{pos}]={layer_id} is out of range "
                        f"[0, {n_layers}) for a model with {n_layers} layers."
                    )
        # Final logit softcapping: matches the reference and bounds logits to
        # (-cap, cap), keeping them finite under float16.
        self.logit_softcapping = text_config.final_logit_softcapping

    def input_tree(self) -> OrderedDict[str, object]:
        """Names the language graph's inputs; see :func:`language_input_tree`."""
        return language_input_tree(
            self.devices, self._embedding_dtype, self.dim, self.kv_params
        )

    def trace(
        self,
        *input_types: Type[Any],
        name: str | None = None,
        module: GraphModule | None = None,
        custom_extensions: Iterable[Path] = (),
        is_device_graph: bool = False,
    ) -> Graph:
        """Traces the language graph, whose inputs are :meth:`input_tree`'s
        leaves, then on a pipeline stage after the first the hidden states
        per device the previous stage returned.

        Adapts the flat graph inputs to :meth:`__call__`: stages the row
        offsets on every device and splits the KV cache into its sliding and
        global collections.

        Args:
            *input_types: The graph's input types, the leaves of
                :meth:`input_tree`.
            name: The name of the graph. Defaults to the class's qualified
                name.
            module: The MLIR module to add the graph to. Defaults to a new
                module.
            custom_extensions: Paths to custom kernel libraries the graph
                uses.
            is_device_graph: Whether to build a device graph.

        Returns:
            The traced :class:`~max.graph.Graph`.
        """
        # The KV cache comes last, and unflattens through the KV params.
        head = OrderedDict(
            (key, types)
            for key, types in self.input_tree().items()
            if key != "kv"
        )
        head_treedef = tree.flatten(head)[1]

        def forward(*inputs: Value[Any]) -> DecoderOutputs:
            remaining = iter(inputs)
            named = tree.unflatten(head_treedef, remaining, exact=False)
            assert isinstance(named, OrderedDict)
            batch, vision = named["batch"], named["vision"]
            signal_buffers = [v.buffer for v in named["signals"]["signal"]]
            sliding_kv, global_kv = self.kv_params.unflatten_basic_kv_tree(
                remaining
            )
            return self(
                tokens=batch["tokens"].tensor,
                signal_buffers=signal_buffers,
                sliding_kv_collections=sliding_kv,
                global_kv_collections=global_kv,
                return_n_logits=batch["return_n_logits"].tensor,
                # Staged once on the first device. A collective rather than
                # a transfer per device: those sync across streams, which
                # breaks graph capture.
                input_row_offsets=ops.distributed_broadcast(
                    batch["row_offsets"].tensor, signal_buffers
                ),
                image_embeddings=[v.tensor for v in vision["image_embeddings"]],
                image_token_indices=[
                    v.tensor for v in vision["image_token_indices"]
                ],
                boundary_inputs=(
                    None if self.holds_first else [v.tensor for v in remaining]
                ),
            )

        return Graph(
            type(self).__qualname__ if name is None else name,
            forward,
            input_types=input_types,
            custom_extensions=list(custom_extensions),
            module=module,
            is_device_graph=is_device_graph,
        )

    def __call__(
        self,
        tokens: TensorValue,
        signal_buffers: Sequence[BufferValue],
        sliding_kv_collections: Sequence[PagedCacheValues],
        global_kv_collections: Sequence[PagedCacheValues],
        return_n_logits: TensorValue,
        input_row_offsets: Sequence[TensorValue],
        image_embeddings: Sequence[TensorValue],
        image_token_indices: Sequence[TensorValue],
        *,
        boundary_inputs: Sequence[TensorValue] | None = None,
        **kwargs: object,
    ) -> DecoderOutputs:
        """Runs this model's decoder layers.

        A pipeline stage after the first starts from ``boundary_inputs``,
        the hidden states per device the previous stage returned, and
        ignores the tokens and image inputs. A stage before the last returns
        its hidden states per device instead of logits.
        """
        kv_collections_by_type = {
            "sliding_attention": sliding_kv_collections,
            "full_attention": global_kv_collections,
        }

        if self.holds_first:
            h = [
                merge_multimodal_embeddings(
                    inputs_embeds=h_device,
                    multimodal_embeddings=img_embed,
                    image_token_indices=img_tok_indices,
                )
                for h_device, img_embed, img_tok_indices in zip(
                    self.embed_tokens(tokens, signal_buffers),
                    image_embeddings,
                    image_token_indices,
                    strict=True,
                )
            ]
        else:
            assert boundary_inputs is not None, (
                "A pipeline stage after the first takes the previous stage's"
                " hidden states."
            )
            h = list(boundary_inputs)

        capture_layer_set: set[int] | None = None
        capture_hidden_states: list[list[TensorValue]] | None = None
        if (
            self.target_layer_ids is not None
            and self.return_hidden_states == ReturnHiddenStates.SELECTED_LAYERS
        ):
            capture_layer_set = set(self.target_layer_ids)
            capture_hidden_states = []

        # Run through transformer layers
        for idx in self.layer_ids:
            layer_idx_tensor = ops.constant(
                idx, DType.uint32, device=self.devices[0]
            )
            kv_collections = kv_collections_by_type[self._layer_kv_key[idx]]
            h = self.layers[idx](
                layer_idx_tensor,
                h,
                signal_buffers,
                kv_collections,
                input_row_offsets=input_row_offsets,
                **kwargs,
            )
            if (
                capture_layer_set is not None
                and capture_hidden_states is not None
                and idx in capture_layer_set
            ):
                capture_hidden_states.append(list(h))

        if not self.holds_last:
            return DecoderOutputs(
                hidden_states=HiddenStates(stage_boundary=tuple(h))
            )
        return self._postprocess_logits(
            h,
            input_row_offsets,
            return_n_logits,
            signal_buffers,
            capture_hidden_states=capture_hidden_states,
        )
