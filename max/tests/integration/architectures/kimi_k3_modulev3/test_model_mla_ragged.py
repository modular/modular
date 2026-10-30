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
"""Verifies ``KimiK3Model`` dispatches a real paged cache down to its MLA layer.

An MLA+dense model run padded (empty ``kv_inputs``) and again ragged over a
decompressed cache should produce the same logits; the sublayer arithmetic
is already proven elsewhere (``test_mla_attention_ragged.py``,
``test_decoder_layer_mla_ragged.py``), so this only covers the model's own
``kv_params``/``layer_idx`` construction, prefill-plan sharing, and
``forward``'s ``kv_inputs`` unflattening. The ragged form reconstructs its
cache from raw graph values, so it is compiled rather than called directly.
The deep-stack variant grows the attention-residual candidate count past 2
end to end, which the one-layer model never does.
"""

from __future__ import annotations

import numpy as np
import pytest
from _paged_cache_test_utils import build_paged_cache
from max import tree
from max.driver import CPU, Accelerator
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.legacy.kv_cache import MHAKVCacheParams
from max.nn.legacy.transformer import ReturnLogits
from max.pipelines.architectures.kimi_k3_modulev3.layers.mla_attention import (
    KimiK3MLAAttention,
)
from max.pipelines.architectures.kimi_k3_modulev3.layers.mlp import (
    KimiK3DenseMLP,
)
from max.pipelines.architectures.kimi_k3_modulev3.model import (
    KimiK3Model,
    KimiK3ModelConfig,
)
from max.pipelines.lib.interfaces.batch_processor import (
    modulev3_ragged_kv_symbolic_inputs,
)
from max.tensor import Tensor, default_device

_HIDDEN = 32
_VOCAB = 40
_NUM_HEADS = 4
_QK_NOPE_HEAD_DIM = 8
_QK_ROPE_HEAD_DIM = 4
_V_HEAD_DIM = 8
_Q_LORA_RANK = 16
_KV_LORA_RANK = 12
_SEQ_LEN = 6
_PAGE_SIZE = 128


def _config(
    num_hidden_layers: int = 1, attn_res_block_size: int = 2
) -> KimiK3ModelConfig:
    return KimiK3ModelConfig(
        hidden_size=_HIDDEN,
        vocab_size=_VOCAB,
        num_hidden_layers=num_hidden_layers,
        kda_layer_indices=[],
        full_attn_layer_indices=list(range(num_hidden_layers)),
        kda_num_heads=1,
        kda_head_dim=32,
        kda_conv_kernel_size=4,
        kda_gate_lower_bound=None,
        num_attention_heads=_NUM_HEADS,
        qk_nope_head_dim=_QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=_QK_ROPE_HEAD_DIM,
        v_head_dim=_V_HEAD_DIM,
        q_lora_rank=_Q_LORA_RANK,
        kv_lora_rank=_KV_LORA_RANK,
        attn_res_block_size=attn_res_block_size,
        first_k_dense_replace=num_hidden_layers,
        intermediate_size=48,
        routed_expert_hidden_size=24,
        moe_intermediate_size=16,
        num_experts=1,
        num_experts_per_token=1,
        num_shared_experts=1,
        rms_norm_eps=1e-5,
        situ_beta=1.5,
        situ_linear_beta=1.5,
    )


def _assign_weights(model: KimiK3Model, rng: np.random.Generator) -> None:
    def bf16(*shape: int) -> Tensor:
        arr = (rng.standard_normal(shape) * 0.02).astype(np.float32)
        return Tensor(arr, dtype=DType.float32).cast(DType.bfloat16)

    def ones(*shape: int) -> Tensor:
        return Tensor(
            np.ones(shape, dtype=np.float32), dtype=DType.float32
        ).cast(DType.bfloat16)

    model.embed_tokens.weight = bf16(_VOCAB, _HIDDEN)
    model.lm_head_weight = bf16(_VOCAB, _HIDDEN)
    model.final_norm_weight = ones(_HIDDEN)
    model.output_attn_res_proj_weight = bf16(1, _HIDDEN)
    model.output_attn_res_norm_weight = ones(_HIDDEN)

    for layer in model.layers:
        layer.input_layernorm_weight = ones(_HIDDEN)
        layer.post_attention_layernorm_weight = ones(_HIDDEN)
        layer.self_attention_res_proj_weight = bf16(1, _HIDDEN)
        layer.self_attention_res_norm_weight = ones(_HIDDEN)
        layer.mlp_res_proj_weight = bf16(1, _HIDDEN)
        layer.mlp_res_norm_weight = ones(_HIDDEN)

        attn = layer.attention
        assert isinstance(attn, KimiK3MLAAttention)
        q_head_dim = _QK_NOPE_HEAD_DIM + _QK_ROPE_HEAD_DIM
        attn.q_a_proj.weight = bf16(_Q_LORA_RANK, _HIDDEN)
        attn.q_a_layernorm.weight = ones(_Q_LORA_RANK)
        attn.q_b_proj.weight = bf16(_NUM_HEADS * q_head_dim, _Q_LORA_RANK)
        attn.kv_a_proj_with_mqa.weight = bf16(
            _KV_LORA_RANK + _QK_ROPE_HEAD_DIM, _HIDDEN
        )
        attn.kv_a_layernorm.weight = ones(_KV_LORA_RANK)
        attn.kv_b_proj.weight = bf16(
            _NUM_HEADS * (_QK_NOPE_HEAD_DIM + _V_HEAD_DIM), _KV_LORA_RANK
        )
        attn.g_proj.weight = bf16(_NUM_HEADS * _V_HEAD_DIM, _HIDDEN)
        attn.o_proj.weight = bf16(_HIDDEN, _NUM_HEADS * _V_HEAD_DIM)

        ffn = layer.ffn
        assert isinstance(ffn, KimiK3DenseMLP)
        ffn.gate_up_proj.weight = bf16(2 * 48, _HIDDEN)
        ffn.down_proj.weight = bf16(_HIDDEN, 48)


def _run_padded_and_ragged(
    config: KimiK3ModelConfig, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Runs the same model padded (empty ``kv_inputs``) and ragged over a
    decompressed cache, with identical weights, and returns both logits."""
    input_ids_np = np.arange(_SEQ_LEN, dtype=np.int64) % _VOCAB
    input_row_offsets_np = np.array([0, _SEQ_LEN], dtype=np.uint32)
    return_n_logits_np = np.array([1], dtype=np.int64)

    device = Accelerator()
    with default_device(device):
        kv_params = MHAKVCacheParams(
            dtype=DType.bfloat16,
            n_kv_heads=_NUM_HEADS,
            head_dim=_QK_NOPE_HEAD_DIM + _QK_ROPE_HEAD_DIM,
            num_layers=config.num_hidden_layers,
            devices=[DeviceRef.from_device(device)],
            page_size=_PAGE_SIZE,
        )

        padded_model = KimiK3Model(
            config, return_logits=ReturnLogits.LAST_TOKEN
        )
        _assign_weights(padded_model, np.random.default_rng(seed))
        padded_outputs = padded_model(
            Tensor(input_ids_np, dtype=DType.int64),
            Tensor(return_n_logits_np, dtype=DType.int64),
            Tensor(input_row_offsets_np, dtype=DType.uint32),
        )
        out_padded = padded_outputs.next_token_logits.cast(
            DType.float32
        ).to_numpy()

        ragged_model = KimiK3Model(config, kv_params=kv_params)
        _assign_weights(ragged_model, np.random.default_rng(seed))
        kv_collection = build_paged_cache(kv_params, [_SEQ_LEN], device)
        run = ragged_model.compile(
            *modulev3_ragged_kv_symbolic_inputs(
                kv_params=kv_params,
                device_refs=[DeviceRef.from_device(device)],
            )
        )
        ragged_outputs = run(
            Tensor(input_ids_np, dtype=DType.int64),
            Tensor(return_n_logits_np, dtype=DType.int64, device=CPU()),
            Tensor(input_row_offsets_np, dtype=DType.uint32),
            *tree.leaves(kv_collection),
        )
        out_ragged = ragged_outputs.next_token_logits.cast(
            DType.float32
        ).to_numpy()
    return out_padded, out_ragged


def _assert_close(
    out_ragged: np.ndarray, out_padded: np.ndarray, what: str
) -> None:
    err = float(
        np.max(np.abs(out_ragged - out_padded))
        / max(np.max(np.abs(out_padded)), 1e-12)
    )
    assert err < 1e-2, f"{what}: ragged MLA vs padded mismatch: {err}"


@pytest.mark.compile_on_gpu(
    "mo.mha.no_cache is staged as an undeclared custom op the interpreter can't bind"
)
def test_model_decompressed_ragged_matches_padded() -> None:
    out_padded, out_ragged = _run_padded_and_ragged(_config(), 20261022)
    _assert_close(out_ragged, out_padded, "model-level")


@pytest.mark.compile_on_gpu(
    "mo.mha.no_cache is staged as an undeclared custom op the interpreter can't bind"
)
def test_model_deep_residual_stack_matches_padded() -> None:
    """Three layers at ``attn_res_block_size`` 1 grow the residual stack
    every layer, so the fused mix-plus-norm op runs end to end at candidate
    counts 2, 3 and 4, plus layer 0's empty-stack plain norm. The published
    model reaches 9 candidates at block size 12; the kernel test sweeps
    that ladder, and this is the end-to-end check that a per-count wiring
    bug cannot pass."""
    config = _config(num_hidden_layers=3, attn_res_block_size=1)
    out_padded, out_ragged = _run_padded_and_ragged(config, 20261023)
    _assert_close(out_ragged, out_padded, "deep-stack")
