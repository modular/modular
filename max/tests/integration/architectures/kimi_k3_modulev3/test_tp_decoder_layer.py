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
"""TP=1 vs TP=2 numerical parity for the assembled ``KimiK3DecoderLayer``.

Sublayers each have their own TP test; this one covers decoder_layer.py's
own assembly (sublayer dispatch, allreduce placement) via two shape combos:
MLA+dense and KDA+MoE.
"""

from __future__ import annotations

import os

os.environ.setdefault("MODULAR_DEBUG", "source-tracebacks")

import numpy as np
import pytest
from max.driver import Accelerator
from max.dtype import DType
from max.experimental.sharding import DeviceMesh, NamedMapping
from max.experimental.tensor import Tensor, default_device
from max.pipelines.architectures.kimi_k3_modulev3.layers.decoder_layer import (
    KimiK3DecoderLayer,
    tensor_parallel_decoder_layer,
)
from max.pipelines.architectures.kimi_k3_modulev3.layers.kda_attention import (
    KimiK3KDAAttention,
)
from max.pipelines.architectures.kimi_k3_modulev3.layers.mla_attention import (
    KimiK3MLAAttention,
)
from max.pipelines.architectures.kimi_k3_modulev3.layers.mlp import (
    KimiK3DenseMLP,
)
from max.pipelines.architectures.kimi_k3_modulev3.layers.moe_latent import (
    KimiK3LatentMoE,
)

_RNG = np.random.default_rng(20261012)

# Layer 1 of block size 2 exercises both the pre- and post-attention
# residual-mix branches, unlike layer 0's degenerate empty-stack case.
_LAYER_IDX = 1
_ATTN_RES_BLOCK_SIZE = 2


def _rel_error(got: np.ndarray, want: np.ndarray) -> float:
    return float(np.max(np.abs(got - want)) / max(np.max(np.abs(want)), 1e-12))


def _arr(*shape: int) -> np.ndarray:
    return (_RNG.standard_normal(shape) * 0.02).astype(np.float32)


@pytest.mark.compile_on_gpu(
    "mo.mha.no_cache is staged as an undeclared custom op the interpreter can't bind"
)
def test_decoder_layer_mla_dense_tp1_matches_tp2() -> None:
    # Matches test_tp_attention.py's MLA parameters: flash_attention_gpu has
    # compiled-shape constraints on head_dim/head count.
    hidden, heads, qk_nope, qk_rope, v_head, q_lora, kv_lora = (
        64,
        8,
        16,
        8,
        16,
        32,
        24,
    )
    inter = 96
    beta, linear_beta = 1.5, 1.5
    q_head_dim = qk_nope + qk_rope
    total_seq_len = 5

    weights = {
        "q_a_proj": _arr(q_lora, hidden),
        "q_b_proj": _arr(heads * q_head_dim, q_lora),
        "kv_a_proj_with_mqa": _arr(kv_lora + qk_rope, hidden),
        "kv_b_proj": _arr(heads * (qk_nope + v_head), kv_lora),
        "g_proj": _arr(heads * v_head, hidden),
        "o_proj": _arr(hidden, heads * v_head),
        # Default RMSNorm weight is bf16 on an Accelerator; force fp32 to match the rest.
        "q_a_layernorm": np.ones(q_lora, dtype=np.float32),
        "kv_a_layernorm": np.ones(kv_lora, dtype=np.float32),
    }
    ffn_weights = {
        "gate_up_proj": _arr(2 * inter, hidden),
        "down_proj": _arr(hidden, inter),
    }
    res_weights = {
        "input_layernorm_weight": np.ones(hidden, dtype=np.float32),
        "post_attention_layernorm_weight": np.ones(hidden, dtype=np.float32),
        "self_attention_res_proj_weight": _arr(1, hidden),
        "self_attention_res_norm_weight": np.ones(hidden, dtype=np.float32),
        "mlp_res_proj_weight": _arr(1, hidden),
        "mlp_res_norm_weight": np.ones(hidden, dtype=np.float32),
    }
    x_np = _arr(total_seq_len, hidden)
    block_residual_np = [_arr(total_seq_len, hidden)]

    def build() -> KimiK3DecoderLayer:
        attention = KimiK3MLAAttention(
            hidden_size=hidden,
            num_heads=heads,
            qk_nope_head_dim=qk_nope,
            qk_rope_head_dim=qk_rope,
            v_head_dim=v_head,
            q_lora_rank=q_lora,
            kv_lora_rank=kv_lora,
        )
        for name, arr in weights.items():
            getattr(attention, name).weight = Tensor(arr, dtype=DType.float32)
        ffn = KimiK3DenseMLP(
            hidden_size=hidden,
            intermediate_size=inter,
            beta=beta,
            linear_beta=linear_beta,
        )
        for name, arr in ffn_weights.items():
            getattr(ffn, name).weight = Tensor(arr, dtype=DType.float32)
        layer = KimiK3DecoderLayer(
            layer_idx=_LAYER_IDX,
            attn_res_block_size=_ATTN_RES_BLOCK_SIZE,
            is_kda=False,
            is_dense=True,
            hidden_size=hidden,
            attention=attention,
            ffn=ffn,
        )
        for name, arr in res_weights.items():
            setattr(layer, name, Tensor(arr, dtype=DType.float32))
        return layer

    with default_device(Accelerator(0)):
        layer1 = build()
        out1, _ = layer1(
            Tensor(x_np, dtype=DType.float32),
            [Tensor(c, dtype=DType.float32) for c in block_residual_np],
        )
        out1_np = out1.to_numpy()

    devices = [Accelerator(0), Accelerator(1)]
    mesh = DeviceMesh(
        devices=tuple(devices), mesh_shape=(2,), axis_names=("tp",)
    )
    with default_device(mesh):
        layer2 = build()
        tensor_parallel_decoder_layer(layer2, mesh)
        out2, _ = layer2(
            Tensor(x_np, dtype=DType.float32),
            [Tensor(c, dtype=DType.float32) for c in block_residual_np],
        )
        out2_np = out2.to_numpy()

    err = _rel_error(out2_np, out1_np)
    assert err < 1e-2, f"MLA+dense decoder layer TP mismatch: {err}"


@pytest.mark.compile_on_gpu("gated_delta_conv1d_fwd mutates its state in place")
def test_decoder_layer_kda_moe_tp1_matches_tp2() -> None:
    # kda_decode only supports compiled head_dim shapes 32 or 128.
    hidden, heads, head_dim, conv_kernel = 64, 8, 32, 4
    inner = heads * head_dim
    latent, moe_inter = 24, 16
    num_experts, top_k = 64, 4
    shared_inter = moe_inter
    beta, linear_beta, eps = 1.5, 1.5, 1e-5

    attn_weights = {
        "q_proj": _arr(inner, hidden),
        "k_proj": _arr(inner, hidden),
        "v_proj": _arr(inner, hidden),
        "f_a_proj": _arr(head_dim, hidden),
        "f_b_proj": _arr(inner, head_dim),
        "b_proj": _arr(heads, hidden),
        "g_proj": _arr(inner, hidden),
        "o_proj": _arr(hidden, inner),
    }
    conv_weight_np = _arr(3 * inner, conv_kernel)
    a_log_np = np.linspace(
        -1.0, 0.5, heads * head_dim, dtype=np.float32
    ).reshape(heads, head_dim)
    dt_bias_np = _arr(heads, head_dim)
    o_norm_np = (1.0 + 0.1 * _RNG.standard_normal(head_dim)).astype(np.float32)

    gate_weight_np = _arr(num_experts, hidden)
    bias_np = np.zeros(num_experts, dtype=np.float32)
    down_proj_np = _arr(latent, hidden)
    up_proj_np = _arr(hidden, latent)
    norm_np = np.ones(latent, dtype=np.float32)
    experts_gate_up_np = _arr(num_experts, 2 * moe_inter, latent)
    experts_down_np = _arr(num_experts, latent, moe_inter)
    shared_gate_up_np = _arr(2 * shared_inter, hidden)
    shared_down_np = _arr(hidden, shared_inter)

    res_weights = {
        "input_layernorm_weight": np.ones(hidden, dtype=np.float32),
        "post_attention_layernorm_weight": np.ones(hidden, dtype=np.float32),
        "self_attention_res_proj_weight": _arr(1, hidden),
        "self_attention_res_norm_weight": np.ones(hidden, dtype=np.float32),
        "mlp_res_proj_weight": _arr(1, hidden),
        "mlp_res_norm_weight": np.ones(hidden, dtype=np.float32),
    }

    total_seq_len, batch_size, max_slots = 6, 2, 2
    x_np = _arr(total_seq_len, hidden)
    block_residual_np = [_arr(total_seq_len, hidden)]
    input_row_offsets_np = np.array(
        [i * (total_seq_len // batch_size) for i in range(batch_size + 1)],
        dtype=np.uint32,
    )
    slot_idx_np = np.arange(batch_size, dtype=np.uint32)

    def build() -> KimiK3DecoderLayer:
        attention = KimiK3KDAAttention(
            hidden_size=hidden,
            num_heads=heads,
            head_dim=head_dim,
            conv_kernel_size=conv_kernel,
            gate_lower_bound=None,
        )
        for name, arr in attn_weights.items():
            getattr(attention, name).weight = Tensor(arr, dtype=DType.float32)
        attention.conv_weight = Tensor(conv_weight_np, dtype=DType.float32)
        attention.a_log = Tensor(a_log_np, dtype=DType.float32)
        attention.dt_bias = Tensor(dt_bias_np, dtype=DType.float32)
        attention.o_norm.weight = Tensor(o_norm_np, dtype=DType.float32)

        ffn = KimiK3LatentMoE(
            hidden_size=hidden,
            latent_size=latent,
            moe_intermediate_size=moe_inter,
            shared_intermediate_size=shared_inter,
            num_experts=num_experts,
            num_experts_per_token=top_k,
            beta=beta,
            linear_beta=linear_beta,
            rms_norm_eps=eps,
            routed_scaling_factor=1.0,
            num_expert_group=1,
            topk_group=1,
            moe_renormalize=True,
        )
        ffn.gate.gate_score.weight = Tensor(gate_weight_np, dtype=DType.float32)
        ffn.gate.e_score_correction_bias = Tensor(bias_np, dtype=DType.float32)
        ffn.routed_expert_down_proj.weight = Tensor(
            down_proj_np, dtype=DType.float32
        )
        ffn.routed_expert_up_proj.weight = Tensor(
            up_proj_np, dtype=DType.float32
        )
        ffn.routed_expert_norm.weight = Tensor(norm_np, dtype=DType.float32)
        ffn.experts_gate_up_proj = Tensor(
            experts_gate_up_np, dtype=DType.float32
        )
        ffn.experts_down_proj = Tensor(experts_down_np, dtype=DType.float32)
        ffn.shared_experts.gate_up_proj.weight = Tensor(
            shared_gate_up_np, dtype=DType.float32
        )
        ffn.shared_experts.down_proj.weight = Tensor(
            shared_down_np, dtype=DType.float32
        )

        layer = KimiK3DecoderLayer(
            layer_idx=_LAYER_IDX,
            attn_res_block_size=_ATTN_RES_BLOCK_SIZE,
            is_kda=True,
            is_dense=False,
            hidden_size=hidden,
            attention=attention,
            ffn=ffn,
        )
        for name, arr in res_weights.items():
            setattr(layer, name, Tensor(arr, dtype=DType.float32))
        return layer

    with default_device(Accelerator(0)):
        layer1 = build()
        conv_pool1 = Tensor.zeros(
            [max_slots, 3 * inner, conv_kernel - 1], dtype=DType.float32
        )
        state_pool1 = Tensor.zeros(
            [max_slots, heads, head_dim, head_dim], dtype=DType.float32
        )
        out1, _ = layer1(
            Tensor(x_np, dtype=DType.float32),
            [Tensor(c, dtype=DType.float32) for c in block_residual_np],
            conv_pool=conv_pool1,
            conv_row_id=Tensor(slot_idx_np, dtype=DType.uint32),
            state_pool=state_pool1,
            recurrent_row_id=Tensor(slot_idx_np, dtype=DType.uint32),
            input_row_offsets=Tensor(input_row_offsets_np, dtype=DType.uint32),
        )
        out1_np = out1.to_numpy()

    devices = [Accelerator(0), Accelerator(1)]
    mesh = DeviceMesh(
        devices=tuple(devices), mesh_shape=(2,), axis_names=("tp",)
    )
    with default_device(mesh):
        layer2 = build()
        tensor_parallel_decoder_layer(layer2, mesh)
        pool_mapping = NamedMapping(mesh, (None, "tp"))
        conv_pool2 = Tensor.zeros(
            [max_slots, 3 * inner, conv_kernel - 1],
            dtype=DType.float32,
            device=pool_mapping,
        )
        state_pool2 = Tensor.zeros(
            [max_slots, heads, head_dim, head_dim],
            dtype=DType.float32,
            device=pool_mapping,
        )
        out2, _ = layer2(
            Tensor(x_np, dtype=DType.float32),
            [Tensor(c, dtype=DType.float32) for c in block_residual_np],
            conv_pool=conv_pool2,
            conv_row_id=Tensor(slot_idx_np, dtype=DType.uint32),
            state_pool=state_pool2,
            recurrent_row_id=Tensor(slot_idx_np, dtype=DType.uint32),
            input_row_offsets=Tensor(input_row_offsets_np, dtype=DType.uint32),
        )
        out2_np = out2.to_numpy()

    err = _rel_error(out2_np, out1_np)
    assert err < 1e-2, f"KDA+MoE decoder layer TP mismatch: {err}"
