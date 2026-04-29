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

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import (
    BufferValue,
    DeviceRef,
    Graph,
    TensorType,
    TensorValue,
    ops,
)
from max.interfaces import RequestID, TokenBuffer
from max.kv_cache import PagedKVCacheManager
from max.nn.attention import MHAMaskVariant
from max.nn.comm.allreduce import Signals
from max.nn.kernels import (
    KVCacheParams,
    flash_attention_ragged,
    fused_qk_ragged_rope,
    fused_qkv_ragged_matmul,
    rms_norm_key_cache,
)
from max.nn.kv_cache import PagedCacheValues, unflatten_ragged_attention_inputs
from max.nn.linear import MLP, linear
from max.nn.rotary_embedding import Llama3RotaryEmbedding
from max.nn.transformer.distributed_transformer import forward_sharded_layers
from max.pipelines.architectures.gemma3.layers.attention import (
    Gemma3Attention as MaxGemma3Attention,
)
from max.pipelines.architectures.gemma3.layers.rms_norm import (
    Gemma3RMSNorm,
    gemma3_rms_norm_fused_residual_add,
)
from max.pipelines.architectures.gemma3.layers.transformer_block import (
    Gemma3TransformerBlock as MaxGemma3TransformerBlock,
)
from max.pipelines.core import TextContext
from torch.utils.dlpack import from_dlpack

PAGE_SIZE = 128
EPS = 1e-6
DEFAULT_LAYER_IDX = 0
DEFAULT_LAYER_TYPE = "local"
DEFAULT_BASELINE_VARIANT = "full"
DEFAULT_PRODUCER_VARIANT = "mlp"
COMPILE_ONLY_RESIDUAL_LADDER_MODE = "residual_ladder_pre_ffn_norm"
BASELINE_VARIANTS = (
    "full",
    "attention_only",
    "residual_only",
    "post_attention_residual_only",
    "post_mlp_residual_only",
)
PRODUCER_VARIANTS = (
    "mlp",
    "pre_ffn_norm",
    "down_proj_only",
    "gate_up_activation_only",
    "materialized_mlp",
    "permuted_mlp",
    "hidden_permuted_mlp",
    "upstream_permuted_mlp",
    "native_upstream_permuted_mlp",
)
Q_NORM_STD = 0.68
K_NORM_STD = 0.793
Q_PROJ_STD = 0.0284
K_PROJ_STD = 0.0309
V_PROJ_STD = 0.0309
O_PROJ_STD = 0.0237
RMS_NORM_STD = 0.05
MLP_GATE_STD = 0.018
MLP_UP_STD = 0.018
MLP_DOWN_STD = 0.015
WARMUP_ITERS = 20
TIMED_ITERS = 50
CACHE_LEN_BASE = 1024
CACHE_LEN_STEP = 7
BATCH_SIZES = (64, 128)
MAX_EXTRA_STEPS = WARMUP_ITERS + TIMED_ITERS


def _resolve_layer_metadata(config: dict[str, Any]) -> tuple[int, str]:
    layer_idx = int(
        os.environ.get(
            "PROFILE_TRANSFORMER_BLOCK_LAYER_IDX", str(DEFAULT_LAYER_IDX)
        )
    )
    num_hidden_layers = int(config["num_hidden_layers"])
    assert 0 <= layer_idx < num_hidden_layers, (
        f"layer_idx={layer_idx} must be in [0, {num_hidden_layers})"
    )
    layer_type = (
        "local"
        if bool((layer_idx + 1) % config["sliding_window_pattern"])
        else "global"
    )
    expected_layer_type = os.environ.get(
        "PROFILE_TRANSFORMER_BLOCK_LAYER_TYPE", DEFAULT_LAYER_TYPE
    )
    assert layer_type == expected_layer_type, (
        f"expected {expected_layer_type} layer, got {layer_type} for layer_idx={layer_idx}"
    )
    return layer_idx, layer_type


def _resolve_kv_num_layers(config: dict[str, Any], layer_idx: int) -> int:
    min_num_layers = layer_idx + 1
    kv_num_layers = int(
        os.environ.get(
            "PROFILE_TRANSFORMER_BLOCK_KV_NUM_LAYERS", str(min_num_layers)
        )
    )
    assert kv_num_layers >= min_num_layers, (
        f"kv_num_layers={kv_num_layers} must cover layer_idx={layer_idx}"
    )
    assert kv_num_layers <= int(config["num_hidden_layers"]), (
        "kv_num_layers cannot exceed the model layer count "
        f"({config['num_hidden_layers']})"
    )
    return kv_num_layers


def _resolve_hidden_activation(config: dict[str, Any]) -> str:
    hidden_activation = str(config["hidden_activation"])
    return {
        "gelu_pytorch_tanh": "gelu_tanh",
        "swish": "silu",
    }.get(hidden_activation, hidden_activation)


def _resolve_baseline_variant() -> str:
    baseline_variant = os.environ.get(
        "PROFILE_TRANSFORMER_BLOCK_BASELINE_VARIANT",
        DEFAULT_BASELINE_VARIANT,
    )
    assert baseline_variant in BASELINE_VARIANTS, (
        "baseline variant must be one of "
        f"{BASELINE_VARIANTS}, got {baseline_variant!r}"
    )
    return baseline_variant


def _resolve_producer_variant() -> str:
    producer_variant = os.environ.get(
        "PROFILE_TRANSFORMER_BLOCK_PRODUCER_VARIANT",
        DEFAULT_PRODUCER_VARIANT,
    )
    assert producer_variant in PRODUCER_VARIANTS, (
        "producer variant must be one of "
        f"{PRODUCER_VARIANTS}, got {producer_variant!r}"
    )
    return producer_variant


def _resolve_compile_only_mode() -> str | None:
    compile_only_mode = os.environ.get(
        "PROFILE_TRANSFORMER_BLOCK_COMPILE_ONLY", ""
    ).strip()
    if not compile_only_mode:
        return None
    assert compile_only_mode == COMPILE_ONLY_RESIDUAL_LADDER_MODE, (
        "compile-only mode must be "
        f"{COMPILE_ONLY_RESIDUAL_LADDER_MODE!r}, got "
        f"{compile_only_mode!r}"
    )
    return compile_only_mode


def _baseline_variant_flags(
    baseline_variant: str,
) -> tuple[bool, bool, bool]:
    match baseline_variant:
        case "full":
            return False, False, False
        case "attention_only":
            return False, True, True
        case "residual_only":
            return True, False, False
        case "post_attention_residual_only":
            return True, False, True
        case "post_mlp_residual_only":
            return True, True, False
        case _:
            raise ValueError(
                f"unsupported baseline variant: {baseline_variant}"
            )


def _residual_variant_name(
    *,
    use_fused_post_attention_residual_add: bool,
    use_fused_post_mlp_residual_add: bool,
) -> str:
    match (
        use_fused_post_attention_residual_add,
        use_fused_post_mlp_residual_add,
    ):
        case (True, True):
            return "FusedResidual"
        case (False, False):
            return "BaselineResidual"
        case (False, True):
            return "BaselinePostAttentionFusedPostMlpResidual"
        case (True, False):
            return "FusedPostAttentionBaselinePostMlpResidual"
        case _:
            raise ValueError("unsupported residual variant")


def _graph_variant_name(
    *,
    layer_type: str,
    use_fused_attention: bool,
    use_fused_post_attention_residual_add: bool,
    use_fused_post_mlp_residual_add: bool,
    producer_variant: str,
) -> str:
    attention_variant = (
        "FusedAttention" if use_fused_attention else "BaselineAttention"
    )
    residual_variant = _residual_variant_name(
        use_fused_post_attention_residual_add=(
            use_fused_post_attention_residual_add
        ),
        use_fused_post_mlp_residual_add=use_fused_post_mlp_residual_add,
    )
    match producer_variant:
        case "mlp":
            producer_variant_name = "MlpProducer"
        case "pre_ffn_norm":
            producer_variant_name = "PreFfnNormProducer"
        case "down_proj_only":
            producer_variant_name = "DownProjOnlyProducer"
        case "gate_up_activation_only":
            producer_variant_name = "GateUpActivationOnlyProducer"
        case "materialized_mlp":
            producer_variant_name = "MaterializedMlpProducer"
        case "permuted_mlp":
            producer_variant_name = "PermutedMlpProducer"
        case "hidden_permuted_mlp":
            producer_variant_name = "HiddenPermutedMlpProducer"
        case "upstream_permuted_mlp":
            producer_variant_name = "UpstreamPermutedMlpProducer"
        case "native_upstream_permuted_mlp":
            producer_variant_name = "NativeUpstreamPermutedMlpProducer"
        case _:
            raise ValueError(
                f"unsupported producer variant: {producer_variant}"
            )
    return (
        f"Gemma3TransformerBlockDecode{layer_type.title()}"
        f"{attention_variant}{residual_variant}{producer_variant_name}"
    )


def _load_text_config() -> dict[str, Any]:
    config_path = Path(os.environ["PIPELINES_TESTDATA"]) / "config.json"
    with open(config_path) as file:
        data = json.load(file)
    return data.get("text_config", data)


def _make_weight_registry(config: dict[str, Any]) -> dict[str, torch.Tensor]:
    torch.manual_seed(42)
    q_dim = config["head_dim"] * config["num_attention_heads"]
    kv_dim = config["head_dim"] * config["num_key_value_heads"]
    hidden_size = config["hidden_size"]
    intermediate_size = config["intermediate_size"]
    return {
        "block.self_attn.k_norm.weight": (
            torch.randn(config["head_dim"], dtype=torch.bfloat16) * K_NORM_STD
        ),
        "block.self_attn.k_proj.weight": (
            torch.randn(kv_dim, hidden_size, dtype=torch.bfloat16) * K_PROJ_STD
        ),
        "block.self_attn.o_proj.weight": (
            torch.randn(hidden_size, q_dim, dtype=torch.bfloat16) * O_PROJ_STD
        ),
        "block.self_attn.q_norm.weight": (
            torch.randn(config["head_dim"], dtype=torch.bfloat16) * Q_NORM_STD
        ),
        "block.self_attn.q_proj.weight": (
            torch.randn(q_dim, hidden_size, dtype=torch.bfloat16) * Q_PROJ_STD
        ),
        "block.self_attn.v_proj.weight": (
            torch.randn(kv_dim, hidden_size, dtype=torch.bfloat16) * V_PROJ_STD
        ),
        "block.mlp.gate_proj.weight": (
            torch.randn(intermediate_size, hidden_size, dtype=torch.bfloat16)
            * MLP_GATE_STD
        ),
        "block.mlp.down_proj.weight": (
            torch.randn(hidden_size, intermediate_size, dtype=torch.bfloat16)
            * MLP_DOWN_STD
        ),
        "block.mlp.up_proj.weight": (
            torch.randn(intermediate_size, hidden_size, dtype=torch.bfloat16)
            * MLP_UP_STD
        ),
        "block.input_layernorm.weight": (
            torch.randn(hidden_size, dtype=torch.bfloat16) * RMS_NORM_STD
        ),
        "block.post_attention_layernorm.weight": (
            torch.randn(hidden_size, dtype=torch.bfloat16) * RMS_NORM_STD
        ),
        "block.pre_feedforward_layernorm.weight": (
            torch.randn(hidden_size, dtype=torch.bfloat16) * RMS_NORM_STD
        ),
        "block.post_feedforward_layernorm.weight": (
            torch.randn(hidden_size, dtype=torch.bfloat16) * RMS_NORM_STD
        ),
        "next_input_layernorm.weight": (
            torch.randn(hidden_size, dtype=torch.bfloat16) * RMS_NORM_STD
        ),
    }


def _feed_forward_chunk_permutation_from_dims(
    hidden_size: int,
    intermediate_size: int,
) -> tuple[int, ...]:
    repeat_factor, remainder = divmod(intermediate_size, hidden_size)
    assert remainder == 0, (
        "feed-forward chunk permutation requires the intermediate size to "
        f"be an integer multiple of hidden size, got {intermediate_size} "
        f"and {hidden_size}"
    )
    if repeat_factor == 1:
        return (0,)
    permutation = tuple(range(1, repeat_factor, 2)) + tuple(
        range(0, repeat_factor, 2)
    )
    assert permutation != tuple(range(repeat_factor))
    return permutation


def _permute_feed_forward_torch_tensor(
    tensor: torch.Tensor,
    *,
    hidden_size: int,
    intermediate_size: int,
    axis: int,
) -> torch.Tensor:
    permutation = _feed_forward_chunk_permutation_from_dims(
        hidden_size, intermediate_size
    )
    if len(permutation) == 1:
        return tensor
    tensor_chunks = torch.split(tensor, hidden_size, dim=axis)
    assert len(tensor_chunks) == len(permutation)
    return torch.cat(
        [tensor_chunks[idx] for idx in permutation],
        dim=axis,
    ).contiguous()


def _materialize_weight_registry_for_producer_variant(
    weight_registry: dict[str, torch.Tensor],
    *,
    config: dict[str, Any],
    producer_variant: str,
) -> tuple[dict[str, torch.Tensor], str]:
    if producer_variant != "native_upstream_permuted_mlp":
        return weight_registry, producer_variant

    hidden_size = int(config["hidden_size"])
    intermediate_size = int(config["intermediate_size"])
    materialized_registry = dict(weight_registry)
    materialized_registry["block.mlp.gate_proj.weight"] = (
        _permute_feed_forward_torch_tensor(
            weight_registry["block.mlp.gate_proj.weight"],
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            axis=0,
        )
    )
    materialized_registry["block.mlp.up_proj.weight"] = (
        _permute_feed_forward_torch_tensor(
            weight_registry["block.mlp.up_proj.weight"],
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            axis=0,
        )
    )
    materialized_registry["block.mlp.down_proj.weight"] = (
        _permute_feed_forward_torch_tensor(
            weight_registry["block.mlp.down_proj.weight"],
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            axis=1,
        )
    )
    return materialized_registry, "mlp"


class _Gemma3AttentionBaseline(MaxGemma3Attention):
    def __call__(
        self,
        x: TensorValue,
        kv_collection: PagedCacheValues,
        **kwargs,
    ) -> TensorValue:
        total_seq_len = x.shape[0]
        layer_idx = ops.constant(
            self.layer_idx, DType.uint32, device=DeviceRef.CPU()
        )

        xq = fused_qkv_ragged_matmul(
            self.kv_params,
            input=x,
            wqkv=self.wqkv,
            bias=self.wqkv_bias,
            input_row_offsets=kwargs["input_row_offsets"],
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            n_heads=self.n_heads,
        )
        xq = xq.reshape((-1, self.n_heads, self.kv_params.head_dim))

        use_local = bool((self.layer_idx + 1) % self.sliding_window_pattern)
        rope = self.rope_local if use_local else self.rope_global
        freqs_cis = ops.cast(rope.freqs_cis, xq.dtype).to(xq.device)

        rms_norm_key_cache(
            self.kv_params,
            kv_collection=kv_collection,
            gamma=self.k_norm.weight.cast(self.kv_params.dtype).to(
                self.devices[0]
            ),
            epsilon=self.qk_norm_eps,
            layer_idx=layer_idx,
            total_seq_len=total_seq_len,
            input_row_offsets=kwargs["input_row_offsets"],
            weight_offset=1.0,
        )
        xq = self.q_norm(xq)
        xq = fused_qk_ragged_rope(
            self.kv_params,
            xq,
            kwargs["input_row_offsets"],
            kv_collection,
            freqs_cis,
            layer_idx,
            interleaved=rope.interleaved,
        )

        mask_variant = (
            MHAMaskVariant.SLIDING_WINDOW_CAUSAL_MASK
            if use_local
            else MHAMaskVariant.CAUSAL_MASK
        )
        attn_out = flash_attention_ragged(
            self.kv_params,
            input=xq,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            input_row_offsets=kwargs["input_row_offsets"],
            mask_variant=mask_variant,
            scale=self.scale,
            local_window_size=self.local_window_size,
        )
        attn_out = ops.reshape(attn_out, shape=[total_seq_len, -1])
        return self.o_proj(attn_out)


class _Gemma3TransformerBlockVariant(MaxGemma3TransformerBlock):
    def __init__(
        self,
        *,
        use_fused_post_attention_residual_add: bool,
        use_fused_post_mlp_residual_add: bool,
        producer_variant: str,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.use_fused_post_attention_residual_add = (
            use_fused_post_attention_residual_add
        )
        self.use_fused_post_mlp_residual_add = use_fused_post_mlp_residual_add
        self.producer_variant = producer_variant

    def _make_down_proj_only_input(
        self,
        x: TensorValue,
        mlp_shard: MLP,
    ) -> TensorValue:
        repeat_factor, remainder = divmod(
            mlp_shard.feed_forward_length, mlp_shard.hidden_dim
        )
        assert remainder == 0, (
            "down_proj_only producer requires the feed-forward width to be "
            f"an integer multiple of hidden_dim, got "
            f"{mlp_shard.feed_forward_length} and {mlp_shard.hidden_dim}"
        )
        if repeat_factor == 1:
            return x
        return ops.concat([x] * repeat_factor, axis=1)

    def _project_feed_forward_hidden_to_model_dim(
        self,
        hidden: TensorValue,
        mlp_shard: MLP,
    ) -> TensorValue:
        repeat_factor, remainder = divmod(
            mlp_shard.feed_forward_length, mlp_shard.hidden_dim
        )
        assert remainder == 0, (
            "gate_up_activation_only producer requires the feed-forward "
            "width to be an integer multiple of hidden_dim, got "
            f"{mlp_shard.feed_forward_length} and {mlp_shard.hidden_dim}"
        )
        if repeat_factor == 1:
            return hidden
        hidden = ops.reshape(
            hidden,
            [hidden.shape[0], repeat_factor, mlp_shard.hidden_dim],
        )
        hidden = ops.sum(hidden, axis=1)
        return ops.reshape(hidden, [hidden.shape[0], mlp_shard.hidden_dim])

    def _make_gate_up_activation_hidden(
        self,
        x: TensorValue,
        mlp_shard: MLP,
    ) -> TensorValue:
        if not mlp_shard._can_used_fused_mlp():
            return mlp_shard.activation_function(
                mlp_shard.gate_proj(x)
            ) * mlp_shard.up_proj(x)

        output = linear(
            x,
            mlp_shard._concat_or_max_gate_up_weights(),
            mlp_shard.quantization_encoding,
            mlp_shard.quant_config,
            input_scale=mlp_shard._concat_or_max_gate_up_input_scale(),
            weight_scale=mlp_shard._concat_or_max_gate_up_scales(),
            weight_scale_2=mlp_shard._concat_or_max_gate_up_weight_scale_2(),
        )
        bias = mlp_shard._concat_or_max_gate_up_bias()
        if bias is not None:
            output += bias

        feed_forward_length = mlp_shard.gate_proj.weight.shape[0]
        gate_out, up_out = ops.split(
            output, [feed_forward_length, feed_forward_length], axis=1
        )
        return mlp_shard.activation_function(gate_out) * up_out

    def _feed_forward_chunk_permutation(
        self,
        mlp_shard: MLP,
    ) -> tuple[int, ...]:
        repeat_factor, remainder = divmod(
            mlp_shard.feed_forward_length, mlp_shard.hidden_dim
        )
        assert remainder == 0, (
            "permuted_mlp producer requires the feed-forward width to be "
            f"an integer multiple of hidden_dim, got "
            f"{mlp_shard.feed_forward_length} and {mlp_shard.hidden_dim}"
        )
        if repeat_factor == 1:
            return (0,)
        permutation = tuple(range(1, repeat_factor, 2)) + tuple(
            range(0, repeat_factor, 2)
        )
        assert permutation != tuple(range(repeat_factor))
        return permutation

    def _permute_feed_forward_tensor(
        self,
        tensor: TensorValue,
        mlp_shard: MLP,
        *,
        axis: int,
    ) -> TensorValue:
        permutation = self._feed_forward_chunk_permutation(mlp_shard)
        if len(permutation) == 1:
            return tensor
        tensor_chunks = ops.split(
            tensor,
            [mlp_shard.hidden_dim] * len(permutation),
            axis=axis,
        )
        return ops.concat(
            [tensor_chunks[idx] for idx in permutation],
            axis=axis,
        )

    def _permute_feed_forward_hidden(
        self,
        hidden: TensorValue,
        mlp_shard: MLP,
    ) -> TensorValue:
        return self._permute_feed_forward_tensor(hidden, mlp_shard, axis=1)

    def _make_permuted_down_proj_weight(
        self,
        mlp_shard: MLP,
        device: DeviceRef,
    ) -> TensorValue:
        weight = mlp_shard.down_proj.weight.to(device)
        return self._permute_feed_forward_tensor(weight, mlp_shard, axis=1)

    def _make_permuted_gate_up_tensor(
        self,
        gate_tensor: TensorValue | None,
        up_tensor: TensorValue | None,
        mlp_shard: MLP,
        device: DeviceRef,
    ) -> TensorValue | None:
        if gate_tensor is None or up_tensor is None:
            return None
        if len(gate_tensor.shape) != 0:
            gate_tensor = self._permute_feed_forward_tensor(
                gate_tensor.to(device), mlp_shard, axis=0
            )
            up_tensor = self._permute_feed_forward_tensor(
                up_tensor.to(device), mlp_shard, axis=0
            )
        return mlp_shard._concat_or_max_gate_up_tensors(gate_tensor, up_tensor)

    def _materialize_feed_forward_hidden(
        self,
        hidden: TensorValue,
        mlp_shard: MLP,
    ) -> TensorValue:
        repeat_factor, remainder = divmod(
            mlp_shard.feed_forward_length, mlp_shard.hidden_dim
        )
        assert remainder == 0, (
            "materialized_mlp producer requires the feed-forward width to be "
            f"an integer multiple of hidden_dim, got "
            f"{mlp_shard.feed_forward_length} and {mlp_shard.hidden_dim}"
        )
        if repeat_factor == 1:
            return hidden
        hidden_chunks = ops.split(
            hidden,
            [mlp_shard.hidden_dim] * repeat_factor,
            axis=1,
        )
        return ops.concat(hidden_chunks, axis=1)

    def _make_gate_up_activation_only_output(
        self,
        x: TensorValue,
        mlp_shard: MLP,
    ) -> TensorValue:
        hidden = self._make_gate_up_activation_hidden(x, mlp_shard)
        return self._project_feed_forward_hidden_to_model_dim(hidden, mlp_shard)

    def _make_materialized_mlp_output(
        self,
        x: TensorValue,
        mlp_shard: MLP,
    ) -> TensorValue:
        hidden = self._make_gate_up_activation_hidden(x, mlp_shard)
        hidden = self._materialize_feed_forward_hidden(hidden, mlp_shard)
        return mlp_shard.down_proj(hidden)

    def _make_permuted_mlp_output(
        self,
        x: TensorValue,
        mlp_shard: MLP,
    ) -> TensorValue:
        hidden = self._make_gate_up_activation_hidden(x, mlp_shard)
        hidden = self._permute_feed_forward_hidden(hidden, mlp_shard)
        weight = self._make_permuted_down_proj_weight(mlp_shard, hidden.device)
        output = linear(
            hidden,
            weight,
            mlp_shard.down_proj.weight.quantization_encoding,
            mlp_shard.down_proj.quant_config,
            mlp_shard.down_proj.input_scale,
            mlp_shard.down_proj.weight_scale,
            mlp_shard.down_proj.weight_scale_2,
        )
        if mlp_shard.down_proj.bias is not None:
            output += mlp_shard.down_proj.bias.to(output.device)
        return output

    def _make_hidden_permuted_mlp_output(
        self,
        x: TensorValue,
        mlp_shard: MLP,
    ) -> TensorValue:
        hidden = self._make_gate_up_activation_hidden(x, mlp_shard)
        hidden = self._permute_feed_forward_hidden(hidden, mlp_shard)
        return mlp_shard.down_proj(hidden)

    def _make_upstream_permuted_mlp_output(
        self,
        x: TensorValue,
        mlp_shard: MLP,
    ) -> TensorValue:
        if not mlp_shard._can_used_fused_mlp():
            hidden = self._make_gate_up_activation_hidden(x, mlp_shard)
            hidden = self._permute_feed_forward_hidden(hidden, mlp_shard)
        else:
            # Emit the odd-even hidden order directly from gate/up so
            # `down_proj` sees round 89's layout without a runtime permute.
            weight = self._make_permuted_gate_up_tensor(
                mlp_shard.gate_proj.weight,
                mlp_shard.up_proj.weight,
                mlp_shard,
                x.device,
            )
            assert weight is not None
            output = linear(
                x,
                weight,
                mlp_shard.quantization_encoding,
                mlp_shard.quant_config,
                input_scale=self._make_permuted_gate_up_tensor(
                    mlp_shard.gate_proj.input_scale,
                    mlp_shard.up_proj.input_scale,
                    mlp_shard,
                    x.device,
                ),
                weight_scale=self._make_permuted_gate_up_tensor(
                    mlp_shard.gate_proj.weight_scale,
                    mlp_shard.up_proj.weight_scale,
                    mlp_shard,
                    x.device,
                ),
                weight_scale_2=self._make_permuted_gate_up_tensor(
                    mlp_shard.gate_proj.weight_scale_2,
                    mlp_shard.up_proj.weight_scale_2,
                    mlp_shard,
                    x.device,
                ),
            )
            bias = self._make_permuted_gate_up_tensor(
                mlp_shard.gate_proj.bias,
                mlp_shard.up_proj.bias,
                mlp_shard,
                x.device,
            )
            if bias is not None:
                output += bias

            feed_forward_length = mlp_shard.gate_proj.weight.shape[0]
            gate_out, up_out = ops.split(
                output,
                [feed_forward_length, feed_forward_length],
                axis=1,
            )
            hidden = mlp_shard.activation_function(gate_out) * up_out

        weight = self._make_permuted_down_proj_weight(mlp_shard, hidden.device)
        output = linear(
            hidden,
            weight,
            mlp_shard.down_proj.weight.quantization_encoding,
            mlp_shard.down_proj.quant_config,
            mlp_shard.down_proj.input_scale,
            mlp_shard.down_proj.weight_scale,
            mlp_shard.down_proj.weight_scale_2,
        )
        if mlp_shard.down_proj.bias is not None:
            output += mlp_shard.down_proj.bias.to(output.device)
        return output

    def __call__(
        self,
        layer_idx: TensorValue,
        xs: list[TensorValue],
        signal_buffers: list[BufferValue],
        kv_collections: list[PagedCacheValues],
        input_row_offsets: list[TensorValue],
        normalized_xs: Sequence[TensorValue] | None = None,
        next_input_layernorm_shards: Sequence[Gemma3RMSNorm] | None = None,
        **kwargs,
    ) -> tuple[list[TensorValue], list[TensorValue] | None]:
        del layer_idx
        residual = xs
        norm_xs = (
            list(normalized_xs)
            if normalized_xs is not None
            else forward_sharded_layers(self.input_layernorm_shards, xs)
        )
        attn_out = [
            shard(
                norm_xs[i],
                kv_collections[i],
                input_row_offsets=input_row_offsets[i],
                **kwargs,
            )
            for i, shard in enumerate(self.self_attn_shards)
        ]
        attn_out = self.allreduce(attn_out, signal_buffers)

        if self.use_fused_post_attention_residual_add:
            fused_attn_norm = [
                gemma3_rms_norm_fused_residual_add(
                    attn_out[i],
                    residual[i],
                    cast(
                        Gemma3RMSNorm, self.post_attention_layernorm_shards[i]
                    ),
                    cast(
                        Gemma3RMSNorm,
                        self.pre_feedforward_layernorm_shards[i],
                    ),
                )
                for i in range(len(attn_out))
            ]
            norm_xs = [fused_output for fused_output, _ in fused_attn_norm]
            residual = [fused_residual for _, fused_residual in fused_attn_norm]
        else:
            norm_xs = forward_sharded_layers(
                self.post_attention_layernorm_shards, attn_out
            )
            residual = [residual[i] + norm_xs[i] for i in range(len(norm_xs))]
            norm_xs = forward_sharded_layers(
                self.pre_feedforward_layernorm_shards, residual
            )

        if self.producer_variant == "mlp":
            hidden_states = forward_sharded_layers(self.mlp_shards, norm_xs)
            hidden_states = self.allreduce(hidden_states, signal_buffers)
        elif self.producer_variant == "pre_ffn_norm":
            hidden_states = list(norm_xs)
        elif self.producer_variant == "down_proj_only":
            hidden_states = []
            for i, norm_x in enumerate(norm_xs):
                mlp_shard = cast(MLP, self.mlp_shards[i])
                synthetic_hidden = self._make_down_proj_only_input(
                    norm_x, mlp_shard
                )
                hidden_states.append(mlp_shard.down_proj(synthetic_hidden))
            hidden_states = self.allreduce(hidden_states, signal_buffers)
        elif self.producer_variant == "gate_up_activation_only":
            hidden_states = []
            for i, norm_x in enumerate(norm_xs):
                mlp_shard = cast(MLP, self.mlp_shards[i])
                hidden_states.append(
                    self._make_gate_up_activation_only_output(norm_x, mlp_shard)
                )
            hidden_states = self.allreduce(hidden_states, signal_buffers)
        elif self.producer_variant == "materialized_mlp":
            hidden_states = []
            for i, norm_x in enumerate(norm_xs):
                mlp_shard = cast(MLP, self.mlp_shards[i])
                hidden_states.append(
                    self._make_materialized_mlp_output(norm_x, mlp_shard)
                )
            hidden_states = self.allreduce(hidden_states, signal_buffers)
        elif self.producer_variant == "permuted_mlp":
            hidden_states = []
            for i, norm_x in enumerate(norm_xs):
                mlp_shard = cast(MLP, self.mlp_shards[i])
                hidden_states.append(
                    self._make_permuted_mlp_output(norm_x, mlp_shard)
                )
            hidden_states = self.allreduce(hidden_states, signal_buffers)
        elif self.producer_variant == "hidden_permuted_mlp":
            hidden_states = []
            for i, norm_x in enumerate(norm_xs):
                mlp_shard = cast(MLP, self.mlp_shards[i])
                hidden_states.append(
                    self._make_hidden_permuted_mlp_output(norm_x, mlp_shard)
                )
            hidden_states = self.allreduce(hidden_states, signal_buffers)
        elif self.producer_variant == "upstream_permuted_mlp":
            hidden_states = []
            for i, norm_x in enumerate(norm_xs):
                mlp_shard = cast(MLP, self.mlp_shards[i])
                hidden_states.append(
                    self._make_upstream_permuted_mlp_output(norm_x, mlp_shard)
                )
            hidden_states = self.allreduce(hidden_states, signal_buffers)
        else:
            raise ValueError(
                f"unsupported producer variant: {self.producer_variant}"
            )

        if next_input_layernorm_shards is None:
            hidden_states = forward_sharded_layers(
                self.post_feedforward_layernorm_shards, hidden_states
            )
            return (
                [
                    residual[i] + hidden_states[i]
                    for i in range(len(hidden_states))
                ],
                None,
            )

        if self.use_fused_post_mlp_residual_add:
            fused_mlp_norm = [
                gemma3_rms_norm_fused_residual_add(
                    hidden_states[i],
                    residual[i],
                    cast(
                        Gemma3RMSNorm,
                        self.post_feedforward_layernorm_shards[i],
                    ),
                    cast(Gemma3RMSNorm, next_input_layernorm_shards[i]),
                )
                for i in range(len(hidden_states))
            ]
            return (
                [fused_residual for _, fused_residual in fused_mlp_norm],
                [fused_output for fused_output, _ in fused_mlp_norm],
            )

        hidden_states = forward_sharded_layers(
            self.post_feedforward_layernorm_shards, hidden_states
        )
        residual = [
            residual[i] + hidden_states[i] for i in range(len(hidden_states))
        ]
        next_norm_xs = forward_sharded_layers(
            next_input_layernorm_shards, residual
        )
        return residual, next_norm_xs


class _Gemma3TransformerBlockDecodeHarness:
    def __init__(
        self,
        *,
        kv_params: KVCacheParams,
        config: dict[str, Any],
        use_fused_attention: bool,
        use_fused_post_attention_residual_add: bool,
        use_fused_post_mlp_residual_add: bool,
        producer_variant: str,
        layer_idx: int,
    ) -> None:
        device_ref = DeviceRef.GPU()
        max_seq_len = (
            CACHE_LEN_BASE
            + CACHE_LEN_STEP * (max(BATCH_SIZES) - 1)
            + MAX_EXTRA_STEPS
            + 256
        )
        attention_cls = (
            MaxGemma3Attention
            if use_fused_attention
            else _Gemma3AttentionBaseline
        )
        attention = attention_cls(
            rope_global=Llama3RotaryEmbedding(
                config["hidden_size"],
                config["num_attention_heads"],
                config["rope_theta"],
                max_seq_len,
                interleaved=False,
                head_dim=config["head_dim"],
            ),
            rope_local=Llama3RotaryEmbedding(
                config["hidden_size"],
                config["num_attention_heads"],
                config["rope_local_base_freq"],
                max_seq_len,
                interleaved=False,
                head_dim=config["head_dim"],
            ),
            num_attention_heads=config["num_attention_heads"],
            num_key_value_heads=config["num_key_value_heads"],
            hidden_size=config["hidden_size"],
            kv_params=kv_params,
            layer_idx=layer_idx,
            dtype=DType.bfloat16,
            devices=[device_ref],
            qk_norm_eps=EPS,
            sliding_window_pattern=config["sliding_window_pattern"],
            local_window_size=config["sliding_window"],
            has_bias=bool(config["attention_bias"]),
        )
        self.block = _Gemma3TransformerBlockVariant(
            attention=attention,
            mlp=MLP(
                dtype=DType.bfloat16,
                quantization_encoding=None,
                hidden_dim=config["hidden_size"],
                feed_forward_length=config["intermediate_size"],
                devices=[device_ref],
                activation_function=_resolve_hidden_activation(config),
                has_bias=False,
            ),
            input_layernorm=Gemma3RMSNorm(
                config["hidden_size"], DType.bfloat16, EPS
            ),
            post_attention_layernorm=Gemma3RMSNorm(
                config["hidden_size"], DType.bfloat16, EPS
            ),
            pre_feedforward_layernorm=Gemma3RMSNorm(
                config["hidden_size"], DType.bfloat16, EPS
            ),
            post_feedforward_layernorm=Gemma3RMSNorm(
                config["hidden_size"], DType.bfloat16, EPS
            ),
            devices=[device_ref],
            use_fused_post_attention_residual_add=(
                use_fused_post_attention_residual_add
            ),
            use_fused_post_mlp_residual_add=(use_fused_post_mlp_residual_add),
            producer_variant=producer_variant,
        )
        self.layer_idx = layer_idx
        self.next_input_layernorm = Gemma3RMSNorm(
            config["hidden_size"], DType.bfloat16, EPS
        )
        sharding_strategy = self.block.input_layernorm.sharding_strategy
        assert sharding_strategy is not None
        self.next_input_layernorm.sharding_strategy = sharding_strategy
        self.next_input_layernorm_shards = self.next_input_layernorm.shard(
            [device_ref]
        )

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.block.load_state_dict(
            {
                key.removeprefix("block."): value
                for key, value in state_dict.items()
                if key.startswith("block.")
            }
        )
        self.next_input_layernorm.load_state_dict(
            {"weight": state_dict["next_input_layernorm.weight"]}
        )

    def state_dict(self) -> dict[str, Any]:
        weights = {
            f"block.{key}": value
            for key, value in self.block.state_dict().items()
        }
        weights["next_input_layernorm.weight"] = (
            self.next_input_layernorm.state_dict()["weight"]
        )
        return weights

    def graph_weights(self) -> dict[str, Any]:
        weights = dict(self.block.state_dict())
        weights["weight"] = self.next_input_layernorm.state_dict()["weight"]
        return weights

    def __call__(
        self,
        x: TensorValue,
        normalized_x: TensorValue,
        signal_buffers: list[BufferValue],
        kv_collection: PagedCacheValues,
        input_row_offsets: TensorValue,
    ) -> tuple[TensorValue, TensorValue]:
        hidden_states, next_norm_xs = self.block(
            ops.constant(self.layer_idx, DType.uint32, device=DeviceRef.CPU()),
            [x],
            signal_buffers,
            [kv_collection],
            input_row_offsets=[input_row_offsets],
            normalized_xs=[normalized_x],
            next_input_layernorm_shards=self.next_input_layernorm_shards,
        )
        assert next_norm_xs is not None
        return hidden_states[0], next_norm_xs[0]


def _build_graph(
    *,
    session: InferenceSession,
    kv_params: KVCacheParams,
    config: dict[str, Any],
    weight_registry: dict[str, torch.Tensor],
    use_fused_attention: bool,
    use_fused_post_attention_residual_add: bool,
    use_fused_post_mlp_residual_add: bool,
    producer_variant: str,
    layer_idx: int,
    layer_type: str,
) -> Any:
    device_ref = DeviceRef.GPU()
    materialized_weight_registry, block_producer_variant = (
        _materialize_weight_registry_for_producer_variant(
            weight_registry,
            config=config,
            producer_variant=producer_variant,
        )
    )
    harness = _Gemma3TransformerBlockDecodeHarness(
        kv_params=kv_params,
        config=config,
        use_fused_attention=use_fused_attention,
        use_fused_post_attention_residual_add=(
            use_fused_post_attention_residual_add
        ),
        use_fused_post_mlp_residual_add=(use_fused_post_mlp_residual_add),
        producer_variant=block_producer_variant,
        layer_idx=layer_idx,
    )
    harness.load_state_dict(materialized_weight_registry)
    signals = Signals(devices=[device_ref])

    input_type = TensorType(
        DType.bfloat16,
        ["total_seq_len", config["hidden_size"]],
        device=device_ref,
    )
    normalized_input_type = TensorType(
        DType.bfloat16,
        ["total_seq_len", config["hidden_size"]],
        device=device_ref,
    )
    input_row_offsets_type = TensorType(
        DType.uint32,
        ["input_row_offsets_len"],
        device=device_ref,
    )
    flattened_kv_types = kv_params.get_symbolic_inputs().flatten()

    graph_name = _graph_variant_name(
        layer_type=layer_type,
        use_fused_attention=use_fused_attention,
        use_fused_post_attention_residual_add=(
            use_fused_post_attention_residual_add
        ),
        use_fused_post_mlp_residual_add=use_fused_post_mlp_residual_add,
        producer_variant=producer_variant,
    )

    with Graph(
        graph_name,
        input_types=(
            input_type,
            normalized_input_type,
            input_row_offsets_type,
            *signals.input_types(),
            *flattened_kv_types,
        ),
    ) as graph:
        x, normalized_x, input_row_offsets, signal, *kv_cache = graph.inputs
        kv_collection = unflatten_ragged_attention_inputs(
            kv_cache, n_devices=1
        )[0]
        hidden_states, next_norm = harness(
            x.tensor,
            normalized_x.tensor,
            [signal.buffer],
            kv_collection,
            input_row_offsets.tensor,
        )
        graph.output(hidden_states, next_norm)

    return session.load(graph, weights_registry=harness.graph_weights())


def _run_compile_only_smoke(
    *,
    session: InferenceSession,
    kv_params: KVCacheParams,
    config: dict[str, Any],
    weight_registry: dict[str, torch.Tensor],
    layer_idx: int,
    layer_type: str,
    compile_only_mode: str,
) -> None:
    assert compile_only_mode == COMPILE_ONLY_RESIDUAL_LADDER_MODE
    smoke_cases = (
        {
            "name": "FusedAttentionBaselineResidualPreFfnNorm",
            "use_fused_attention": True,
            "use_fused_post_attention_residual_add": False,
            "use_fused_post_mlp_residual_add": False,
            "producer_variant": "pre_ffn_norm",
        },
        {
            "name": (
                "FusedAttentionFusedPostAttentionBaselinePostMlpResidualPreFfnNorm"
            ),
            "use_fused_attention": True,
            "use_fused_post_attention_residual_add": True,
            "use_fused_post_mlp_residual_add": False,
            "producer_variant": "pre_ffn_norm",
        },
        {
            "name": "FusedAttentionFusedResidualPreFfnNorm",
            "use_fused_attention": True,
            "use_fused_post_attention_residual_add": True,
            "use_fused_post_mlp_residual_add": True,
            "producer_variant": "pre_ffn_norm",
        },
    )

    results: dict[str, Any] = {
        "benchmark_config": {
            "dtype": "bfloat16",
            "hidden_size": config["hidden_size"],
            "intermediate_size": config["intermediate_size"],
            "head_dim": config["head_dim"],
            "num_q_heads": config["num_attention_heads"],
            "num_kv_heads": config["num_key_value_heads"],
            "page_size": PAGE_SIZE,
            "layer_idx": layer_idx,
            "layer_type": layer_type,
            "kv_num_layers": kv_params.num_layers,
            "mode": compile_only_mode,
        },
        "cases": [],
        "status": "pass",
    }

    for case in smoke_cases:
        case_result = {
            "name": case["name"],
            "graph_name": _graph_variant_name(
                layer_type=layer_type,
                use_fused_attention=bool(case["use_fused_attention"]),
                use_fused_post_attention_residual_add=bool(
                    case["use_fused_post_attention_residual_add"]
                ),
                use_fused_post_mlp_residual_add=bool(
                    case["use_fused_post_mlp_residual_add"]
                ),
                producer_variant=str(case["producer_variant"]),
            ),
            "producer_variant": case["producer_variant"],
            "use_fused_attention": case["use_fused_attention"],
            "use_fused_post_attention_residual_add": case[
                "use_fused_post_attention_residual_add"
            ],
            "use_fused_post_mlp_residual_add": case[
                "use_fused_post_mlp_residual_add"
            ],
        }
        try:
            compiled = _build_graph(
                session=session,
                kv_params=kv_params,
                config=config,
                weight_registry=weight_registry,
                use_fused_attention=bool(case["use_fused_attention"]),
                use_fused_post_attention_residual_add=bool(
                    case["use_fused_post_attention_residual_add"]
                ),
                use_fused_post_mlp_residual_add=bool(
                    case["use_fused_post_mlp_residual_add"]
                ),
                producer_variant=str(case["producer_variant"]),
                layer_idx=layer_idx,
                layer_type=layer_type,
            )
        except Exception as exc:
            case_result["status"] = "failed"
            case_result["error"] = str(exc)
            results["cases"].append(case_result)
            results["status"] = "blocker"
            results["first_failure"] = case["name"]
            print(json.dumps(results, indent=2, sort_keys=True))
            raise

        case_result["status"] = "pass"
        results["cases"].append(case_result)
        del compiled
        torch.cuda.empty_cache()

    print(json.dumps(results, indent=2, sort_keys=True))


def _make_text_context(length: int, max_length: int) -> TextContext:
    return TextContext(
        request_id=RequestID(),
        max_length=max_length,
        tokens=TokenBuffer(np.zeros(length, dtype=np.int64)),
    )


def _make_runtime_inputs(
    *,
    session: InferenceSession,
    kv_params: KVCacheParams,
    batch_size: int,
) -> tuple[np.ndarray, Any]:
    cache_lengths = np.asarray(
        [
            CACHE_LEN_BASE + CACHE_LEN_STEP * request_idx
            for request_idx in range(batch_size)
        ],
        dtype=np.uint32,
    )
    max_cache_length = int(cache_lengths[-1]) + MAX_EXTRA_STEPS + 1
    total_num_pages = sum(
        math.ceil((int(cache_length) + MAX_EXTRA_STEPS + 1) / PAGE_SIZE)
        for cache_length in cache_lengths
    )

    kv_manager = PagedKVCacheManager(
        params=kv_params,
        total_num_pages=total_num_pages,
        session=session,
        max_batch_size=batch_size,
    )

    contexts: list[TextContext] = []
    for cache_length in cache_lengths:
        context = _make_text_context(int(cache_length), max_cache_length)
        kv_manager.claim(context.request_id, replica_idx=0)
        kv_manager.alloc(
            context,
            replica_idx=0,
            num_steps=MAX_EXTRA_STEPS + 1,
        )
        contexts.append(context)

    runtime_inputs = kv_manager.runtime_inputs([contexts], num_steps=1).inputs[
        0
    ]
    return cache_lengths, runtime_inputs


def _clone_kv_blocks(blocks: Buffer, seed: int) -> Buffer:
    torch.manual_seed(seed)
    shape = tuple(int(dim) for dim in blocks.shape)
    tensor = torch.randn(
        shape, dtype=torch.bfloat16, device="cuda"
    ).contiguous()
    return Buffer.from_dlpack(tensor)


def _device_uint32_buffer(array: np.ndarray, device: Accelerator) -> Buffer:
    return Buffer.from_numpy(array).to(device)


def _make_benchmark_args(
    *,
    batch_size: int,
    hidden_size: int,
    cache_lengths: np.ndarray,
    runtime_inputs: Any,
    device: Accelerator,
    x_seed: int,
    blocks_seed: int,
) -> tuple[list[tuple[Any, ...]], Buffer]:
    torch.manual_seed(x_seed)
    x = torch.randn(
        (batch_size, hidden_size),
        dtype=torch.bfloat16,
        device="cuda",
    ).contiguous()
    x_buffer = Buffer.from_dlpack(x)
    torch.manual_seed(x_seed + 97)
    normalized_x = torch.randn(
        (batch_size, hidden_size),
        dtype=torch.bfloat16,
        device="cuda",
    ).contiguous()
    normalized_x_buffer = Buffer.from_dlpack(normalized_x)
    row_offsets = _device_uint32_buffer(
        np.arange(batch_size + 1, dtype=np.uint32),
        device,
    )
    signal_buffer = Signals(devices=[DeviceRef.GPU()]).buffers()[0]
    kv_blocks = _clone_kv_blocks(runtime_inputs.blocks, seed=blocks_seed)
    lookup_table = runtime_inputs.lookup_table.to(device)
    dispatch_metadata = runtime_inputs.attention_dispatch_metadata
    assert dispatch_metadata is not None

    args: list[tuple[Any, ...]] = []
    for step in range(MAX_EXTRA_STEPS + 1):
        step_cache_lengths = _device_uint32_buffer(
            cache_lengths + np.uint32(step), device
        )
        args.append(
            (
                x_buffer,
                normalized_x_buffer,
                row_offsets,
                signal_buffer,
                kv_blocks,
                step_cache_lengths,
                lookup_table,
                runtime_inputs.max_lengths,
                dispatch_metadata,
            )
        )
    return args, kv_blocks


def _run_correctness_check(
    *,
    baseline: Any,
    fused: Any,
    baseline_args: tuple[Any, ...],
    fused_args: tuple[Any, ...],
    layer_idx: int,
) -> None:
    baseline_hidden, baseline_next_norm = baseline.execute(*baseline_args)
    fused_hidden, fused_next_norm = fused.execute(*fused_args)
    torch.cuda.synchronize()

    torch.testing.assert_close(
        from_dlpack(baseline_hidden).to(torch.bfloat16),
        from_dlpack(fused_hidden).to(torch.bfloat16),
        rtol=2 * torch.finfo(torch.bfloat16).eps,
        atol=8 * torch.finfo(torch.bfloat16).eps,
    )
    torch.testing.assert_close(
        from_dlpack(baseline_next_norm).to(torch.bfloat16),
        from_dlpack(fused_next_norm).to(torch.bfloat16),
        rtol=2 * torch.finfo(torch.bfloat16).eps,
        atol=8 * torch.finfo(torch.bfloat16).eps,
    )

    baseline_blocks = from_dlpack(baseline_args[4]).to(torch.bfloat16)[
        :, :, layer_idx : layer_idx + 1, :, :, :
    ]
    fused_blocks = from_dlpack(fused_args[4]).to(torch.bfloat16)[
        :, :, layer_idx : layer_idx + 1, :, :, :
    ]
    torch.testing.assert_close(
        baseline_blocks,
        fused_blocks,
        rtol=2 * torch.finfo(torch.bfloat16).eps,
        atol=8 * torch.finfo(torch.bfloat16).eps,
    )


def _benchmark_us(compiled: Any, args: list[tuple[Any, ...]]) -> float:
    warmup_args = args[:WARMUP_ITERS]
    timed_args = args[WARMUP_ITERS : WARMUP_ITERS + TIMED_ITERS]

    for run_args in warmup_args:
        compiled.execute(*run_args)
    torch.cuda.synchronize()

    start_s = time.perf_counter()
    for run_args in timed_args:
        compiled.execute(*run_args)
    torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - start_s
    return elapsed_s * 1e6 / len(timed_args)


def _run_nvidia_smi_query(query_fields: str) -> list[list[str]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--query-{query_fields}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [
        [field.strip() for field in row]
        for row in csv.reader(result.stdout.splitlines())
        if row
    ]


def _build_gpu_isolation_guard() -> dict[str, Any] | None:
    raw_guard = os.environ.get(
        "PROFILE_TRANSFORMER_BLOCK_ENFORCE_GPU_ISOLATION", ""
    ).strip()
    if raw_guard.lower() not in ("1", "true", "yes"):
        return None

    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    target_gpu_index = (
        0
        if cuda_visible_devices == ""
        else int(cuda_visible_devices.split(",")[0])
    )

    for index_text, gpu_uuid in _run_nvidia_smi_query("gpu=index,uuid"):
        if int(index_text) == target_gpu_index:
            return {
                "allowed_pid": os.getpid(),
                "target_gpu_index": target_gpu_index,
                "target_gpu_uuid": gpu_uuid,
            }

    raise AssertionError(
        "Could not resolve the target GPU UUID for "
        f"CUDA_VISIBLE_DEVICES={cuda_visible_devices!r}"
    )


def _assert_gpu_isolation(
    gpu_isolation_guard: dict[str, Any] | None,
    *,
    label: str,
) -> None:
    if gpu_isolation_guard is None:
        return

    resident_compute_apps = []
    unexpected_apps = []
    for (
        gpu_uuid,
        pid_text,
        process_name,
        used_gpu_memory_mib,
    ) in _run_nvidia_smi_query(
        "compute-apps=gpu_uuid,pid,process_name,used_gpu_memory"
    ):
        if gpu_uuid != gpu_isolation_guard["target_gpu_uuid"]:
            continue
        resident_app = {
            "gpu_uuid": gpu_uuid,
            "pid": int(pid_text),
            "process_name": process_name,
            "used_gpu_memory_mib": float(used_gpu_memory_mib),
        }
        resident_compute_apps.append(resident_app)
        if resident_app["pid"] != gpu_isolation_guard["allowed_pid"]:
            unexpected_apps.append(resident_app)

    if unexpected_apps:
        raise AssertionError(
            f"{label} at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}: "
            "unexpected compute apps on physical GPU "
            f"{gpu_isolation_guard['target_gpu_index']} "
            f"({gpu_isolation_guard['target_gpu_uuid']}): "
            f"{unexpected_apps}; all resident apps: {resident_compute_apps}"
        )


def _benchmark_us_guarded(
    compiled: Any,
    args: list[tuple[Any, ...]],
    *,
    gpu_isolation_guard: dict[str, Any] | None,
    label: str,
) -> float:
    _assert_gpu_isolation(
        gpu_isolation_guard,
        label=f"{label}:before_warmup",
    )
    warmup_args = args[:WARMUP_ITERS]
    timed_args = args[WARMUP_ITERS : WARMUP_ITERS + TIMED_ITERS]

    for run_args in warmup_args:
        compiled.execute(*run_args)
    torch.cuda.synchronize()

    start_s = time.perf_counter()
    for run_args in timed_args:
        compiled.execute(*run_args)
    torch.cuda.synchronize()
    _assert_gpu_isolation(
        gpu_isolation_guard,
        label=f"{label}:after_timed",
    )
    elapsed_s = time.perf_counter() - start_s
    return elapsed_s * 1e6 / len(timed_args)


def test_profile_transformer_block_decode() -> None:
    config = _load_text_config()
    layer_idx, layer_type = _resolve_layer_metadata(config)
    kv_num_layers = _resolve_kv_num_layers(config, layer_idx)
    baseline_variant = _resolve_baseline_variant()
    producer_variant = _resolve_producer_variant()
    compile_only_mode = _resolve_compile_only_mode()
    gpu_isolation_guard = _build_gpu_isolation_guard()
    (
        baseline_use_fused_attention,
        baseline_use_fused_post_attention_residual_add,
        baseline_use_fused_post_mlp_residual_add,
    ) = _baseline_variant_flags(baseline_variant)
    session = InferenceSession(devices=[Accelerator(0)])
    device = Accelerator(0)
    kv_params = KVCacheParams(
        dtype=DType.bfloat16,
        devices=[DeviceRef.GPU()],
        n_kv_heads=config["num_key_value_heads"],
        head_dim=config["head_dim"],
        num_layers=kv_num_layers,
        page_size=PAGE_SIZE,
    )
    weight_registry = _make_weight_registry(config)
    if compile_only_mode is not None:
        _run_compile_only_smoke(
            session=session,
            kv_params=kv_params,
            config=config,
            weight_registry=weight_registry,
            layer_idx=layer_idx,
            layer_type=layer_type,
            compile_only_mode=compile_only_mode,
        )
        return

    baseline = _build_graph(
        session=session,
        kv_params=kv_params,
        config=config,
        weight_registry=weight_registry,
        use_fused_attention=baseline_use_fused_attention,
        use_fused_post_attention_residual_add=(
            baseline_use_fused_post_attention_residual_add
        ),
        use_fused_post_mlp_residual_add=(
            baseline_use_fused_post_mlp_residual_add
        ),
        producer_variant=producer_variant,
        layer_idx=layer_idx,
        layer_type=layer_type,
    )
    fused = _build_graph(
        session=session,
        kv_params=kv_params,
        config=config,
        weight_registry=weight_registry,
        use_fused_attention=True,
        use_fused_post_attention_residual_add=True,
        use_fused_post_mlp_residual_add=True,
        producer_variant=producer_variant,
        layer_idx=layer_idx,
        layer_type=layer_type,
    )

    benchmark_config: dict[str, Any] = {
        "dtype": "bfloat16",
        "hidden_size": config["hidden_size"],
        "intermediate_size": config["intermediate_size"],
        "head_dim": config["head_dim"],
        "num_q_heads": config["num_attention_heads"],
        "num_kv_heads": config["num_key_value_heads"],
        "page_size": PAGE_SIZE,
        "layer_idx": layer_idx,
        "layer_type": layer_type,
        "kv_num_layers": kv_num_layers,
        "baseline_variant": baseline_variant,
        "producer_variant": producer_variant,
        "mode": "decode-ragged-transformer-block",
        "cache_len_base": CACHE_LEN_BASE,
        "cache_len_step": CACHE_LEN_STEP,
        "warmup_iters": WARMUP_ITERS,
        "timed_iters": TIMED_ITERS,
    }
    if gpu_isolation_guard is not None:
        benchmark_config["gpu_isolation_guard"] = {
            "allowed_pid": gpu_isolation_guard["allowed_pid"],
            "target_gpu_index": gpu_isolation_guard["target_gpu_index"],
            "target_gpu_uuid": gpu_isolation_guard["target_gpu_uuid"],
        }

    results: dict[str, Any] = {
        "benchmark_config": benchmark_config,
        "correctness": "pass",
        "first_sweep_us": {},
        "confirm_sweep_us": {},
        "average_us": {},
        "average_speedup_ratio_vs_baseline": {},
    }

    for batch_size in BATCH_SIZES:
        cache_lengths, runtime_inputs = _make_runtime_inputs(
            session=session,
            kv_params=kv_params,
            batch_size=batch_size,
        )
        run_name = (
            f"decode_{layer_type}_block_layer{layer_idx}_bs{batch_size}_seq1"
            f"_cache{CACHE_LEN_BASE}_step{CACHE_LEN_STEP}"
            + (
                ""
                if producer_variant == DEFAULT_PRODUCER_VARIANT
                else f"_{producer_variant}"
            )
        )

        baseline_args, _ = _make_benchmark_args(
            batch_size=batch_size,
            hidden_size=config["hidden_size"],
            cache_lengths=cache_lengths,
            runtime_inputs=runtime_inputs,
            device=device,
            x_seed=1000 + batch_size,
            blocks_seed=2000 + batch_size,
        )
        fused_args, _ = _make_benchmark_args(
            batch_size=batch_size,
            hidden_size=config["hidden_size"],
            cache_lengths=cache_lengths,
            runtime_inputs=runtime_inputs,
            device=device,
            x_seed=1000 + batch_size,
            blocks_seed=2000 + batch_size,
        )

        _run_correctness_check(
            baseline=baseline,
            fused=fused,
            baseline_args=baseline_args[0],
            fused_args=fused_args[0],
            layer_idx=layer_idx,
        )

        first_baseline_us = _benchmark_us_guarded(
            baseline,
            baseline_args,
            gpu_isolation_guard=gpu_isolation_guard,
            label=f"{run_name}:baseline:first",
        )
        first_fused_us = _benchmark_us_guarded(
            fused,
            fused_args,
            gpu_isolation_guard=gpu_isolation_guard,
            label=f"{run_name}:fused:first",
        )
        results["first_sweep_us"][run_name] = {
            "baseline": first_baseline_us,
            "fused": first_fused_us,
        }

        del baseline_args
        del fused_args

        baseline_args, _ = _make_benchmark_args(
            batch_size=batch_size,
            hidden_size=config["hidden_size"],
            cache_lengths=cache_lengths,
            runtime_inputs=runtime_inputs,
            device=device,
            x_seed=3000 + batch_size,
            blocks_seed=4000 + batch_size,
        )
        fused_args, _ = _make_benchmark_args(
            batch_size=batch_size,
            hidden_size=config["hidden_size"],
            cache_lengths=cache_lengths,
            runtime_inputs=runtime_inputs,
            device=device,
            x_seed=3000 + batch_size,
            blocks_seed=4000 + batch_size,
        )

        confirm_baseline_us = _benchmark_us_guarded(
            baseline,
            baseline_args,
            gpu_isolation_guard=gpu_isolation_guard,
            label=f"{run_name}:baseline:confirm",
        )
        confirm_fused_us = _benchmark_us_guarded(
            fused,
            fused_args,
            gpu_isolation_guard=gpu_isolation_guard,
            label=f"{run_name}:fused:confirm",
        )
        results["confirm_sweep_us"][run_name] = {
            "baseline": confirm_baseline_us,
            "fused": confirm_fused_us,
        }

        average_baseline_us = 0.5 * (first_baseline_us + confirm_baseline_us)
        average_fused_us = 0.5 * (first_fused_us + confirm_fused_us)
        results["average_us"][run_name] = {
            "baseline": average_baseline_us,
            "fused": average_fused_us,
        }
        results["average_speedup_ratio_vs_baseline"][run_name] = (
            average_baseline_us / average_fused_us
        )

        del baseline_args
        del fused_args
        del runtime_inputs
        torch.cuda.empty_cache()

    speedups = list(results["average_speedup_ratio_vs_baseline"].values())
    results["average_geomean_speedup_vs_baseline"] = math.prod(speedups) ** (
        1.0 / len(speedups)
    )
    print(json.dumps(results, indent=2, sort_keys=True))
