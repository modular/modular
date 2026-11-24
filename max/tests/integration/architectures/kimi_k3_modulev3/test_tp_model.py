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
from _tp_model_test_utils import (
    kv_cache_inputs,
    kv_params_only,
    random_weights,
    rel_error,
    tiny_model_config,
)
from max import functional as F
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
from max.pipelines.lib.interfaces.batch_processor import (
    modulev3_ragged_kv_symbolic_inputs,
)
from max.tensor import Tensor, default_device
from max.tensor.sharding import DeviceMesh


def _assign_weights(model: KimiK3Model, raw: dict[str, np.ndarray]) -> None:
    """Assigns each array directly, skipping ``load_kimi_k3_weights``'s
    ``language_model.model`` checkpoint key prefix."""
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
            for name in (
                "q_proj",
                "k_proj",
                "v_proj",
                "g_proj",
                "b_proj",
                "f_a_proj",
                "f_b_proj",
                "o_proj",
                "o_norm",
            ):
                getattr(attn, name).weight = Tensor(
                    raw[f"{ap}.{name}.weight"], dtype=DType.float32
                )
            attn.q_conv1d = Tensor(
                raw[f"{ap}.q_conv1d.weight"], dtype=DType.float32
            )
            attn.k_conv1d = Tensor(
                raw[f"{ap}.k_conv1d.weight"], dtype=DType.float32
            )
            attn.v_conv1d = Tensor(
                raw[f"{ap}.v_conv1d.weight"], dtype=DType.float32
            )
            attn.a_log = Tensor(raw[f"{ap}.A_log"], dtype=DType.float32)
            attn.dt_bias = Tensor(raw[f"{ap}.dt_bias"], dtype=DType.float32)
        else:
            assert isinstance(attn, KimiK3MLAAttention)
            for name in (
                "q_a_proj",
                "kv_a_proj_with_mqa",
                "g_proj",
                "q_a_layernorm",
                "q_b_proj",
                "kv_a_layernorm",
                "kv_b_proj",
                "o_proj",
            ):
                getattr(attn, name).weight = Tensor(
                    raw[f"{ap}.{name}.weight"], dtype=DType.float32
                )

        ffn = layer.ffn
        if layer.is_dense:
            assert isinstance(ffn, KimiK3DenseMLP)
            for name in ("gate_proj", "up_proj", "down_proj"):
                getattr(ffn, name).weight = Tensor(
                    raw[f"{p}.mlp.{name}.weight"], dtype=DType.float32
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
            for e, expert in enumerate(ffn.experts):
                for w in ("w1", "w2", "w3"):
                    projection = getattr(expert, w)
                    projection.weight = Tensor(
                        raw[f"{mp}.experts.{e}.{w}.weight"],
                        dtype=DType.float32,
                        device=projection.weight.device,
                    )
            for name in ("gate_proj", "up_proj", "down_proj"):
                getattr(ffn.shared_experts, name).weight = Tensor(
                    raw[f"{mp}.shared_experts.{name}.weight"],
                    dtype=DType.float32,
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

    # Both graphs compile before either runs or any cache buffer exists, and
    # the weights realize when each compiled model binds them, so a
    # precompile-mefs record run captures both.
    device0 = Accelerator(0)
    with default_device(device0):
        kv_params1 = kv_params_only(config, [device0])
        with F.lazy():
            model1 = KimiK3Model(config, kv_params=kv_params1)
            _assign_weights(model1, raw)
        run1 = model1.compile(
            *modulev3_ragged_kv_symbolic_inputs(
                kv_params=kv_params1,
                device_refs=[DeviceRef.from_device(device0)],
            )
        )

    devices = [Accelerator(0), Accelerator(1)]
    mesh = DeviceMesh(
        devices=tuple(devices), mesh_shape=(2,), axis_names=("tp",)
    )
    with default_device(mesh):
        kv_params2 = kv_params_only(config, devices)
        with F.lazy():
            model2 = KimiK3Model(config, kv_params=kv_params2)
            _assign_weights(model2, raw)
            tensor_parallel_kimi_k3_model(model2, mesh)
        run2 = model2.compile(
            *modulev3_ragged_kv_symbolic_inputs(
                kv_params=kv_params2,
                device_refs=[DeviceRef.from_device(d) for d in devices],
            )
        )

    with default_device(device0):
        outputs1 = run1(
            Tensor(input_ids_np, dtype=DType.int64),
            Tensor(return_n_logits_np, dtype=DType.int64, device=CPU()),
            Tensor(input_row_offsets_np, dtype=DType.uint32),
            *kv_cache_inputs(config, kv_params1, [device0], prompt_lens),
        )
        out1 = outputs1.next_token_logits.to_numpy()

    with default_device(mesh):
        outputs2 = run2(
            Tensor(input_ids_np, dtype=DType.int64, device=devices[0]),
            Tensor(return_n_logits_np, dtype=DType.int64, device=CPU()),
            Tensor(input_row_offsets_np, dtype=DType.uint32, device=devices[0]),
            *kv_cache_inputs(config, kv_params2, devices, prompt_lens),
        )
        out2 = outputs2.next_token_logits.to_numpy()

    err = rel_error(out2, out1)
    assert err < 1e-2, f"full model TP=1 vs TP=2 mismatch: {err}"
