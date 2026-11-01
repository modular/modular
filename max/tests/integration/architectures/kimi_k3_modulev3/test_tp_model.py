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

import numpy as np
import pytest
from _tp_model_test_utils import (
    kv_params_and_inputs,
    random_weights,
    rel_error,
    tiny_model_config,
)
from max.driver import CPU, Accelerator
from max.dtype import DType
from max.graph import DeviceRef
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
    tensor_parallel_kimi_k3_model,
)
from max.pipelines.architectures.kimi_k3_modulev3.weight_adapters import (
    interleave_for_tp,
)
from max.pipelines.lib.interfaces.batch_processor import (
    modulev3_ragged_kv_symbolic_inputs,
)
from max.tensor import Tensor, default_device
from max.tensor.sharding import DeviceMesh


def _assign_weights(
    model: KimiK3Model, raw: dict[str, np.ndarray], tp_size: int
) -> None:
    """Assigns each array directly, skipping ``load_kimi_k3_weights``'s
    ``language_model.model`` checkpoint key prefix.

    Fused weights are interleaved per TP rank, as the weight adapter writes
    them for a ``tp_size``-way TP model.
    """
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
            attn.qkv_proj.weight = Tensor(
                interleave_for_tp(
                    [
                        raw[f"{ap}.q_proj.weight"],
                        raw[f"{ap}.k_proj.weight"],
                        raw[f"{ap}.v_proj.weight"],
                    ],
                    tp_size,
                ),
                dtype=DType.float32,
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
                interleave_for_tp(
                    np.split(raw[f"{ap}.conv_weight"], 3), tp_size
                ),
                dtype=DType.float32,
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
                interleave_for_tp(
                    [
                        raw[f"{p}.mlp.gate_proj.weight"],
                        raw[f"{p}.mlp.up_proj.weight"],
                    ],
                    tp_size,
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
                    interleave_for_tp(
                        [
                            raw[f"{mp}.experts.{e}.w1.weight"],
                            raw[f"{mp}.experts.{e}.w3.weight"],
                        ],
                        tp_size,
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
                interleave_for_tp(
                    [
                        raw[f"{mp}.shared_experts.gate_proj.weight"],
                        raw[f"{mp}.shared_experts.up_proj.weight"],
                    ],
                    tp_size,
                ),
                dtype=DType.float32,
            )
            ffn.shared_experts.down_proj.weight = Tensor(
                raw[f"{mp}.shared_experts.down_proj.weight"],
                dtype=DType.float32,
            )


@pytest.mark.compile_on_gpu(
    "mo.mha.no_cache and gated_delta_conv1d_fwd are both staged as undeclared"
    " custom ops the interpreter can't bind"
)
def test_model_tp1_matches_tp2() -> None:
    hidden, vocab = 32, 40
    config = tiny_model_config(hidden, vocab)
    raw = random_weights(config)

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
        kv_params1, kv_inputs1 = kv_params_and_inputs(
            config, [device0], prompt_lens
        )
        model1 = KimiK3Model(config, kv_params=kv_params1)
        _assign_weights(model1, raw, 1)
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
        kv_params2, kv_inputs2 = kv_params_and_inputs(
            config, devices, prompt_lens
        )
        model2 = KimiK3Model(config, kv_params=kv_params2)
        _assign_weights(model2, raw, len(devices))
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

    err = rel_error(out2, out1)
    assert err < 1e-2, f"full model TP=1 vs TP=2 mismatch: {err}"
