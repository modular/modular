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
"""Single-device vs expert-parallel numerical parity for the assembled
``KimiK3DecoderLayer``.

Needs 2 real GPUs -- see the BUILD target's ``gpu-count`` exec property.
"""

from __future__ import annotations

import os

os.environ.setdefault("MODULAR_DEBUG", "source-tracebacks")

import numpy as np
import pytest
from max.driver import Accelerator
from max.dtype import DType
from max.engine import InferenceSession
from max.pipelines.architectures.kimi_k3_modulev3.layers import ep
from max.pipelines.architectures.kimi_k3_modulev3.layers.decoder_layer import (
    KimiK3DecoderLayer,
    expert_parallel_decoder_layer,
)
from max.pipelines.architectures.kimi_k3_modulev3.layers.mla_attention import (
    KimiK3MLAAttention,
)
from max.pipelines.architectures.kimi_k3_modulev3.layers.moe_latent import (
    KimiK3LatentMoE,
    expert_parallel_config,
)
from max.tensor import Tensor, default_device
from max.tensor.sharding import DeviceMesh

_RNG = np.random.default_rng(20261017)

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
def test_decoder_layer_mla_moe_single_device_matches_ep2() -> None:
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
    q_head_dim = qk_nope + qk_rope
    # Kernel-valid: EP combine's hid_dim must divide by WARP_SIZE *
    # simd_width_of[combine_dtype] -- thousands of elements on this
    # hardware, not the handful a shrunk synthetic shape would use
    # elsewhere in this port's tests.
    latent, moe_inter = 2048, 16
    # Kernel-valid: the MoE router kernel needs num_threads == num_experts
    # to be a whole number of warps.
    num_experts, top_k = 64, 4
    shared_inter = moe_inter
    beta, linear_beta, eps = 1.5, 1.5, 1e-5
    total_seq_len = 5

    attn_weights = {
        "q_a_proj": _arr(q_lora, hidden),
        "q_b_proj": _arr(heads * q_head_dim, q_lora),
        "kv_a_proj_with_mqa": _arr(kv_lora + qk_rope, hidden),
        "kv_b_proj": _arr(heads * (qk_nope + v_head), kv_lora),
        "g_proj": _arr(heads * v_head, hidden),
        "o_proj": _arr(hidden, heads * v_head),
    }
    norm_weights = {
        "q_a_layernorm": np.ones(q_lora, dtype=np.float32),
        "kv_a_layernorm": np.ones(kv_lora, dtype=np.float32),
    }

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
        for name, arr in attn_weights.items():
            getattr(attention, name).weight = Tensor(
                arr, dtype=DType.float32
            ).cast(DType.bfloat16)
        for name, arr in norm_weights.items():
            getattr(attention, name).weight = Tensor(
                arr, dtype=DType.float32
            ).cast(DType.bfloat16)

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
        # The gate stays float32: a bf16 rounding here could flip a close
        # top-k pick, diverging the two sides for a reason that has
        # nothing to do with the exchange this test actually checks.
        ffn.gate.gate_score.weight = Tensor(gate_weight_np, dtype=DType.float32)
        ffn.gate.e_score_correction_bias = Tensor(bias_np, dtype=DType.float32)
        ffn.routed_expert_down_proj.weight = Tensor(
            down_proj_np, dtype=DType.float32
        ).cast(DType.bfloat16)
        ffn.routed_expert_up_proj.weight = Tensor(
            up_proj_np, dtype=DType.float32
        ).cast(DType.bfloat16)
        ffn.routed_expert_norm.weight = Tensor(
            norm_np, dtype=DType.float32
        ).cast(DType.bfloat16)
        ffn.experts_gate_up_proj = Tensor(
            experts_gate_up_np, dtype=DType.float32
        ).cast(DType.bfloat16)
        ffn.experts_down_proj = Tensor(
            experts_down_np, dtype=DType.float32
        ).cast(DType.bfloat16)
        ffn.shared_experts.gate_up_proj.weight = Tensor(
            shared_gate_up_np, dtype=DType.float32
        ).cast(DType.bfloat16)
        ffn.shared_experts.down_proj.weight = Tensor(
            shared_down_np, dtype=DType.float32
        ).cast(DType.bfloat16)

        layer = KimiK3DecoderLayer(
            layer_idx=_LAYER_IDX,
            attn_res_block_size=_ATTN_RES_BLOCK_SIZE,
            is_kda=False,
            is_dense=False,
            hidden_size=hidden,
            attention=attention,
            ffn=ffn,
        )
        for name, arr in res_weights.items():
            setattr(
                layer,
                name,
                Tensor(arr, dtype=DType.float32).cast(DType.bfloat16),
            )
        return layer

    with default_device(Accelerator(0)):
        layer1 = build()
        out1, _ = layer1(
            Tensor(x_np, dtype=DType.float32).cast(DType.bfloat16),
            [
                Tensor(c, dtype=DType.float32).cast(DType.bfloat16)
                for c in block_residual_np
            ],
        )
        out1_np = out1.cast(DType.float32).to_numpy()

    # MLA only rebinds onto `tp` if the tensor is actually distributed, so
    # a pure-EP mesh needs no `tp` axis.
    devices = [Accelerator(0), Accelerator(1)]
    mesh = DeviceMesh(
        devices=tuple(devices), mesh_shape=(2,), axis_names=("ep",)
    )
    session = InferenceSession(devices=devices)
    with default_device(mesh):
        layer2 = build()
        expert_parallel_decoder_layer(layer2, mesh)
        assert isinstance(layer2.ffn, KimiK3LatentMoE)
        config = expert_parallel_config(
            layer2.ffn, mesh, max_tokens_per_rank=64
        )
        buffers = ep.allocate_ep_buffers(config, mesh, session)
        out2, _ = layer2(
            Tensor(x_np, dtype=DType.float32).cast(DType.bfloat16),
            [
                Tensor(c, dtype=DType.float32).cast(DType.bfloat16)
                for c in block_residual_np
            ],
            ep_config=config,
            ep_buffers=buffers,
        )
        out2_np = out2.cast(DType.float32).to_numpy()

    err = _rel_error(out2_np, out1_np)
    assert err < 1e-2, f"MLA+MoE decoder layer EP mismatch: {err}"
