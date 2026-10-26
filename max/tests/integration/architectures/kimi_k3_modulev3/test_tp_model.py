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
"""TP=1 vs TP=2 numerical parity for the full ``KimiK3Model``.

Covers ``tensor_parallel_kimi_k3_model``'s own top-level wiring (sharding
every layer + ``lm_head_weight`` together) via a tiny 2-layer random-weight
model. Not a checkpoint parity test.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from _paged_cache_test_utils import (
    build_kda_state,
    build_paged_cache,
)
from max import tree
from max.driver import CPU, Accelerator
from max.dtype import DType
from max.experimental.sharding import DeviceMesh
from max.experimental.tensor import Tensor, default_device
from max.graph import DeviceRef
from max.nn.kv_cache import (
    MHAKVCacheParams,
    MultiKVCacheParams,
    RecurrentStateParams,
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
from max.pipelines.architectures.kimi_k3_modulev3.model import (
    KimiK3Model,
    KimiK3ModelConfig,
    tensor_parallel_kimi_k3_model,
)
from max.pipelines.architectures.kimi_k3_modulev3.state_cache import (
    ATTN_CACHE_KEY,
    STATE_CACHE_KEY,
    kda_state_regions,
)
from max.pipelines.lib.interfaces.batch_processor import (
    modulev3_ragged_kv_symbolic_inputs,
)

_RNG = np.random.default_rng(20261012)


def _rel_error(got: np.ndarray, want: np.ndarray) -> float:
    return float(np.max(np.abs(got - want)) / max(np.max(np.abs(want)), 1e-12))


def _config(hidden: int, vocab: int) -> KimiK3ModelConfig:
    return KimiK3ModelConfig(
        hidden_size=hidden,
        vocab_size=vocab,
        num_hidden_layers=2,
        # Layer 0: KDA + dense. Layer 1: MLA + MoE. These are the two
        # dispatch combos.
        kda_layer_indices=[0],
        full_attn_layer_indices=[1],
        # kda_decode only supports compiled head_dim shapes 32 or 128.
        kda_num_heads=2,
        kda_head_dim=32,
        kda_conv_kernel_size=4,
        kda_gate_lower_bound=None,
        num_attention_heads=4,
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
        v_head_dim=8,
        q_lora_rank=16,
        kv_lora_rank=12,
        attn_res_block_size=2,
        first_k_dense_replace=1,
        intermediate_size=48,
        routed_expert_hidden_size=24,
        moe_intermediate_size=16,
        # Kernel-valid: the MoE router kernel needs num_threads ==
        # num_experts to be a whole number of warps.
        num_experts=64,
        num_experts_per_token=4,
        num_shared_experts=1,
        rms_norm_eps=1e-5,
        situ_beta=1.5,
        situ_linear_beta=1.5,
    )


def _random_weights(config: KimiK3ModelConfig) -> dict[str, np.ndarray]:
    def arr(*shape: int) -> np.ndarray:
        return (_RNG.standard_normal(shape) * 0.02).astype(np.float32)

    def ones(*shape: int) -> np.ndarray:
        return np.ones(shape, dtype=np.float32)

    hidden = config.hidden_size
    w: dict[str, np.ndarray] = {
        "embed_tokens.weight": arr(config.vocab_size, hidden),
        "lm_head.weight": arr(config.vocab_size, hidden),
        "norm.weight": ones(hidden),
        "output_attn_res_proj.weight": arr(1, hidden),
        "output_attn_res_norm.weight": ones(hidden),
    }
    kda_inner = config.kda_num_heads * config.kda_head_dim
    mla_q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
    mla_inner = config.num_attention_heads * config.v_head_dim

    for i in range(config.num_hidden_layers):
        layer_prefix = f"layers.{i}"
        w[f"{layer_prefix}.input_layernorm.weight"] = ones(hidden)
        w[f"{layer_prefix}.post_attention_layernorm.weight"] = ones(hidden)
        w[f"{layer_prefix}.self_attention_res_proj.weight"] = arr(1, hidden)
        w[f"{layer_prefix}.self_attention_res_norm.weight"] = ones(hidden)
        w[f"{layer_prefix}.mlp_res_proj.weight"] = arr(1, hidden)
        w[f"{layer_prefix}.mlp_res_norm.weight"] = ones(hidden)

        attn_prefix = f"{layer_prefix}.self_attn"
        if i in config.kda_layer_indices:
            w[f"{attn_prefix}.q_proj.weight"] = arr(kda_inner, hidden)
            w[f"{attn_prefix}.k_proj.weight"] = arr(kda_inner, hidden)
            w[f"{attn_prefix}.v_proj.weight"] = arr(kda_inner, hidden)
            w[f"{attn_prefix}.f_a_proj.weight"] = arr(
                config.kda_head_dim, hidden
            )
            w[f"{attn_prefix}.f_b_proj.weight"] = arr(
                kda_inner, config.kda_head_dim
            )
            w[f"{attn_prefix}.b_proj.weight"] = arr(
                config.kda_num_heads, hidden
            )
            w[f"{attn_prefix}.g_proj.weight"] = arr(kda_inner, hidden)
            w[f"{attn_prefix}.o_proj.weight"] = arr(hidden, kda_inner)
            w[f"{attn_prefix}.o_norm.weight"] = ones(config.kda_head_dim)
            w[f"{attn_prefix}.conv_weight"] = arr(
                3 * kda_inner, config.kda_conv_kernel_size
            )
            w[f"{attn_prefix}.a_log"] = np.linspace(
                -1.0,
                0.5,
                config.kda_num_heads * config.kda_head_dim,
                dtype=np.float32,
            ).reshape(config.kda_num_heads, config.kda_head_dim)
            w[f"{attn_prefix}.dt_bias"] = arr(
                config.kda_num_heads, config.kda_head_dim
            )
        else:
            w[f"{attn_prefix}.q_a_proj.weight"] = arr(
                config.q_lora_rank, hidden
            )
            w[f"{attn_prefix}.q_a_layernorm.weight"] = ones(config.q_lora_rank)
            w[f"{attn_prefix}.q_b_proj.weight"] = arr(
                config.num_attention_heads * mla_q_head_dim,
                config.q_lora_rank,
            )
            w[f"{attn_prefix}.kv_a_proj_with_mqa.weight"] = arr(
                config.kv_lora_rank + config.qk_rope_head_dim, hidden
            )
            w[f"{attn_prefix}.kv_a_layernorm.weight"] = ones(
                config.kv_lora_rank
            )
            w[f"{attn_prefix}.kv_b_proj.weight"] = arr(
                config.num_attention_heads
                * (config.qk_nope_head_dim + config.v_head_dim),
                config.kv_lora_rank,
            )
            w[f"{attn_prefix}.g_proj.weight"] = arr(mla_inner, hidden)
            w[f"{attn_prefix}.o_proj.weight"] = arr(hidden, mla_inner)

        if i < config.first_k_dense_replace:
            w[f"{layer_prefix}.mlp.gate_proj.weight"] = arr(
                config.intermediate_size, hidden
            )
            w[f"{layer_prefix}.mlp.up_proj.weight"] = arr(
                config.intermediate_size, hidden
            )
            w[f"{layer_prefix}.mlp.down_proj.weight"] = arr(
                hidden, config.intermediate_size
            )
        else:
            moe_prefix = f"{layer_prefix}.block_sparse_moe"
            w[f"{moe_prefix}.gate.weight"] = arr(config.num_experts, hidden)
            w[f"{moe_prefix}.gate.e_score_correction_bias"] = np.zeros(
                config.num_experts, dtype=np.float32
            )
            w[f"{moe_prefix}.routed_expert_down_proj.weight"] = arr(
                config.routed_expert_hidden_size, hidden
            )
            w[f"{moe_prefix}.routed_expert_up_proj.weight"] = arr(
                hidden, config.routed_expert_hidden_size
            )
            w[f"{moe_prefix}.routed_expert_norm.weight"] = ones(
                config.routed_expert_hidden_size
            )
            for e in range(config.num_experts):
                w[f"{moe_prefix}.experts.{e}.w1.weight"] = arr(
                    config.moe_intermediate_size,
                    config.routed_expert_hidden_size,
                )
                w[f"{moe_prefix}.experts.{e}.w2.weight"] = arr(
                    config.routed_expert_hidden_size,
                    config.moe_intermediate_size,
                )
                w[f"{moe_prefix}.experts.{e}.w3.weight"] = arr(
                    config.moe_intermediate_size,
                    config.routed_expert_hidden_size,
                )
            shared_inter = (
                config.moe_intermediate_size * config.num_shared_experts
            )
            w[f"{moe_prefix}.shared_experts.gate_proj.weight"] = arr(
                shared_inter, hidden
            )
            w[f"{moe_prefix}.shared_experts.up_proj.weight"] = arr(
                shared_inter, hidden
            )
            w[f"{moe_prefix}.shared_experts.down_proj.weight"] = arr(
                hidden, shared_inter
            )
    return w


def _assign_weights(model: KimiK3Model, raw: dict[str, np.ndarray]) -> None:
    """Assigns each array directly, skipping ``load_kimi_k3_weights``'s
    ``language_model.model`` checkpoint key prefix."""
    config = model.config
    model.embed_tokens.weight = Tensor(
        raw["embed_tokens.weight"], dtype=DType.float32
    )
    model.lm_head_weight = Tensor(raw["lm_head.weight"], dtype=DType.float32)
    model.final_norm_weight = Tensor(raw["norm.weight"], dtype=DType.float32)
    model.output_attn_res_proj_weight = Tensor(
        raw["output_attn_res_proj.weight"], dtype=DType.float32
    )
    model.output_attn_res_norm_weight = Tensor(
        raw["output_attn_res_norm.weight"], dtype=DType.float32
    )

    for i, layer in enumerate(model.layers):
        p = f"layers.{i}"
        layer.input_layernorm_weight = Tensor(
            raw[f"{p}.input_layernorm.weight"], dtype=DType.float32
        )
        layer.post_attention_layernorm_weight = Tensor(
            raw[f"{p}.post_attention_layernorm.weight"], dtype=DType.float32
        )
        layer.self_attention_res_proj_weight = Tensor(
            raw[f"{p}.self_attention_res_proj.weight"], dtype=DType.float32
        )
        layer.self_attention_res_norm_weight = Tensor(
            raw[f"{p}.self_attention_res_norm.weight"], dtype=DType.float32
        )
        layer.mlp_res_proj_weight = Tensor(
            raw[f"{p}.mlp_res_proj.weight"], dtype=DType.float32
        )
        layer.mlp_res_norm_weight = Tensor(
            raw[f"{p}.mlp_res_norm.weight"], dtype=DType.float32
        )

        attn = layer.attention
        ap = f"{p}.self_attn"
        if layer.is_kda:
            assert isinstance(attn, KimiK3KDAAttention)
            attn.q_proj.weight = Tensor(
                raw[f"{ap}.q_proj.weight"], dtype=DType.float32
            )
            attn.k_proj.weight = Tensor(
                raw[f"{ap}.k_proj.weight"], dtype=DType.float32
            )
            attn.v_proj.weight = Tensor(
                raw[f"{ap}.v_proj.weight"], dtype=DType.float32
            )
            attn.f_a_proj.weight = Tensor(
                raw[f"{ap}.f_a_proj.weight"], dtype=DType.float32
            )
            attn.f_b_proj.weight = Tensor(
                raw[f"{ap}.f_b_proj.weight"], dtype=DType.float32
            )
            attn.b_proj.weight = Tensor(
                raw[f"{ap}.b_proj.weight"], dtype=DType.float32
            )
            attn.g_proj.weight = Tensor(
                raw[f"{ap}.g_proj.weight"], dtype=DType.float32
            )
            attn.o_proj.weight = Tensor(
                raw[f"{ap}.o_proj.weight"], dtype=DType.float32
            )
            attn.o_norm.weight = Tensor(
                raw[f"{ap}.o_norm.weight"], dtype=DType.float32
            )
            attn.conv_weight = Tensor(
                raw[f"{ap}.conv_weight"], dtype=DType.float32
            )
            attn.a_log = Tensor(raw[f"{ap}.a_log"], dtype=DType.float32)
            attn.dt_bias = Tensor(raw[f"{ap}.dt_bias"], dtype=DType.float32)
        else:
            assert isinstance(attn, KimiK3MLAAttention)
            attn.q_a_proj.weight = Tensor(
                raw[f"{ap}.q_a_proj.weight"], dtype=DType.float32
            )
            attn.q_a_layernorm.weight = Tensor(
                raw[f"{ap}.q_a_layernorm.weight"], dtype=DType.float32
            )
            attn.q_b_proj.weight = Tensor(
                raw[f"{ap}.q_b_proj.weight"], dtype=DType.float32
            )
            attn.kv_a_proj_with_mqa.weight = Tensor(
                raw[f"{ap}.kv_a_proj_with_mqa.weight"], dtype=DType.float32
            )
            attn.kv_a_layernorm.weight = Tensor(
                raw[f"{ap}.kv_a_layernorm.weight"], dtype=DType.float32
            )
            attn.kv_b_proj.weight = Tensor(
                raw[f"{ap}.kv_b_proj.weight"], dtype=DType.float32
            )
            attn.g_proj.weight = Tensor(
                raw[f"{ap}.g_proj.weight"], dtype=DType.float32
            )
            attn.o_proj.weight = Tensor(
                raw[f"{ap}.o_proj.weight"], dtype=DType.float32
            )

        ffn = layer.ffn
        if layer.is_dense:
            assert isinstance(ffn, KimiK3DenseMLP)
            ffn.gate_up_proj.weight = Tensor(
                np.concatenate(
                    [
                        raw[f"{p}.mlp.gate_proj.weight"],
                        raw[f"{p}.mlp.up_proj.weight"],
                    ],
                    axis=0,
                ),
                dtype=DType.float32,
            )
            ffn.down_proj.weight = Tensor(
                raw[f"{p}.mlp.down_proj.weight"], dtype=DType.float32
            )
        else:
            assert isinstance(ffn, KimiK3LatentMoE)
            mp = f"{p}.block_sparse_moe"
            ffn.gate.gate_score.weight = Tensor(
                raw[f"{mp}.gate.weight"], dtype=DType.float32
            )
            ffn.gate.e_score_correction_bias = Tensor(
                raw[f"{mp}.gate.e_score_correction_bias"], dtype=DType.float32
            )
            ffn.routed_expert_down_proj.weight = Tensor(
                raw[f"{mp}.routed_expert_down_proj.weight"], dtype=DType.float32
            )
            ffn.routed_expert_up_proj.weight = Tensor(
                raw[f"{mp}.routed_expert_up_proj.weight"], dtype=DType.float32
            )
            ffn.routed_expert_norm.weight = Tensor(
                raw[f"{mp}.routed_expert_norm.weight"], dtype=DType.float32
            )
            gate_up = np.stack(
                [
                    np.concatenate(
                        [
                            raw[f"{mp}.experts.{e}.w1.weight"],
                            raw[f"{mp}.experts.{e}.w3.weight"],
                        ],
                        axis=0,
                    )
                    for e in range(config.num_experts)
                ],
                axis=0,
            )
            down = np.stack(
                [
                    raw[f"{mp}.experts.{e}.w2.weight"]
                    for e in range(config.num_experts)
                ],
                axis=0,
            )
            ffn.experts_gate_up_proj = Tensor(gate_up, dtype=DType.float32)
            ffn.experts_down_proj = Tensor(down, dtype=DType.float32)
            ffn.shared_experts.gate_up_proj.weight = Tensor(
                np.concatenate(
                    [
                        raw[f"{mp}.shared_experts.gate_proj.weight"],
                        raw[f"{mp}.shared_experts.up_proj.weight"],
                    ],
                    axis=0,
                ),
                dtype=DType.float32,
            )
            ffn.shared_experts.down_proj.weight = Tensor(
                raw[f"{mp}.shared_experts.down_proj.weight"],
                dtype=DType.float32,
            )


def _kv_params_and_inputs(
    config: KimiK3ModelConfig,
    devices: Sequence[Accelerator],
    prompt_lens: list[int],
) -> tuple[MultiKVCacheParams, list[Tensor]]:
    """Builds a real, per-device cache tree (MLA leaf + KDA state) and its
    flattened inputs, for exactly this batch at this device count."""
    tp = len(devices)
    device_refs = [DeviceRef.from_device(d) for d in devices]
    num_kda = len(config.kda_layer_indices)
    kda_inner = config.kda_num_heads * config.kda_head_dim

    attn_params = MHAKVCacheParams(
        dtype=DType.float32,
        n_kv_heads=config.num_attention_heads,
        head_dim=config.qk_nope_head_dim + config.qk_rope_head_dim,
        num_layers=len(config.full_attn_layer_indices),
        devices=device_refs,
        page_size=128,
    )
    state_params = RecurrentStateParams(
        regions=kda_state_regions(
            num_kda_layers=num_kda,
            conv_dim=3 * kda_inner,
            conv_kernel_size=config.kda_conv_kernel_size,
            num_heads=config.kda_num_heads,
            head_dim=config.kda_head_dim,
            conv_dtype=DType.float32,
            recurrent_dtype=DType.float32,
            num_devices=tp,
        ),
        devices=device_refs,
    )
    kv_params = MultiKVCacheParams.from_params(
        {ATTN_CACHE_KEY: attn_params, STATE_CACHE_KEY: state_params}
    )

    attn_per_device = [
        build_paged_cache(attn_params, prompt_lens, d) for d in devices
    ]
    state_per_device = [
        build_kda_state(
            num_kda_layers=num_kda,
            num_requests=len(prompt_lens),
            conv_dim=3 * kda_inner // tp,
            conv_kernel_size=config.kda_conv_kernel_size,
            num_heads=config.kda_num_heads // tp,
            head_dim=config.kda_head_dim,
            conv_dtype=DType.float32,
            recurrent_dtype=DType.float32,
            device=d,
        )
        for d in devices
    ]
    kv_inputs = tree.leaves(
        {ATTN_CACHE_KEY: attn_per_device, STATE_CACHE_KEY: state_per_device}
    )
    return kv_params, kv_inputs


@pytest.mark.compile_on_gpu(
    "mo.mha.no_cache and gated_delta_conv1d_fwd are both staged as undeclared"
    " custom ops the interpreter can't bind"
)
def test_model_tp1_matches_tp2() -> None:
    hidden, vocab = 32, 40
    config = _config(hidden, vocab)
    raw = _random_weights(config)

    total_seq_len, batch_size = 6, 2
    input_ids_np = np.arange(total_seq_len, dtype=np.int64) % vocab
    input_row_offsets_np = np.array(
        [i * (total_seq_len // batch_size) for i in range(batch_size + 1)],
        dtype=np.uint32,
    )
    prompt_lens = [total_seq_len // batch_size] * batch_size
    return_n_logits_np = np.array([1], dtype=np.int64)

    device0 = Accelerator(0)
    with default_device(device0):
        kv_params1, kv_inputs1 = _kv_params_and_inputs(
            config, [device0], prompt_lens
        )
        model1 = KimiK3Model(config, kv_params=kv_params1)
        _assign_weights(model1, raw)
        run1 = model1.compile(
            *modulev3_ragged_kv_symbolic_inputs(
                kv_params=kv_params1,
                device_refs=[DeviceRef.from_device(device0)],
            )
        )
        outputs1 = run1(
            Tensor(input_ids_np, dtype=DType.int64),
            Tensor(return_n_logits_np, dtype=DType.int64, device=CPU()),
            Tensor(input_row_offsets_np, dtype=DType.uint32),
            *kv_inputs1,
        )
        out1 = outputs1.next_token_logits.to_numpy()

    devices = [Accelerator(0), Accelerator(1)]
    mesh = DeviceMesh(
        devices=tuple(devices), mesh_shape=(2,), axis_names=("tp",)
    )
    with default_device(mesh):
        kv_params2, kv_inputs2 = _kv_params_and_inputs(
            config, devices, prompt_lens
        )
        model2 = KimiK3Model(config, kv_params=kv_params2)
        _assign_weights(model2, raw)
        tensor_parallel_kimi_k3_model(model2, mesh)
        run2 = model2.compile(
            *modulev3_ragged_kv_symbolic_inputs(
                kv_params=kv_params2,
                device_refs=[DeviceRef.from_device(d) for d in devices],
            )
        )
        outputs2 = run2(
            Tensor(input_ids_np, dtype=DType.int64, device=devices[0]),
            Tensor(return_n_logits_np, dtype=DType.int64, device=CPU()),
            Tensor(input_row_offsets_np, dtype=DType.uint32, device=devices[0]),
            *kv_inputs2,
        )
        out2 = outputs2.next_token_logits.to_numpy()

    err = _rel_error(out2, out1)
    assert err < 1e-2, f"full model TP=1 vs TP=2 mismatch: {err}"
