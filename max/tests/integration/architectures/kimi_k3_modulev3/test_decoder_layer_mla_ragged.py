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
"""Verifies ``KimiK3DecoderLayer`` dispatches correctly to the ragged MLA forms.

Builds one decoder layer (MLA attention, dense MLP) and runs ``forward()``
padded (no ``kv_collection``, already proven against the reference
checkpoint via the sublayer's own tests), then again with a real paged
cache -- decompressed in one test, compressed-latent in the other -- and
checks the two outputs match. This is the decoder layer's own dispatch logic
under test, not the attention arithmetic again; see ``test_mla_attention_ragged.py``
and ``test_mla_attention_latent_ragged.py`` for that.
"""

from __future__ import annotations

import numpy as np
import pytest
from _paged_cache_test_utils import build_paged_cache
from max.driver import Accelerator
from max.dtype import DType
from max.experimental.tensor import Tensor, default_device
from max.graph import DeviceRef
from max.nn.legacy.kv_cache import MHAKVCacheParams, MLAKVCacheParams
from max.pipelines.architectures.kimi_k3_modulev3.layers.decoder_layer import (
    KimiK3DecoderLayer,
)
from max.pipelines.architectures.kimi_k3_modulev3.layers.mla_attention import (
    KimiK3MLAAttention,
    kimi_k3_mla_prefill_plan,
    kimi_k3_nope_freqs_cis,
)
from max.pipelines.architectures.kimi_k3_modulev3.layers.mlp import (
    KimiK3DenseMLP,
)

_RNG = np.random.default_rng(20261022)

_LAYER_IDX = 1
_ATTN_RES_BLOCK_SIZE = 2
_PAGE_SIZE = 128


def _rel_error(got: np.ndarray, want: np.ndarray) -> float:
    return float(np.max(np.abs(got - want)) / max(np.max(np.abs(want)), 1e-12))


def _arr(*shape: int) -> np.ndarray:
    return (_RNG.standard_normal(shape) * 0.02).astype(np.float32)


def _build_layer(
    *,
    hidden: int,
    heads: int,
    qk_nope: int,
    qk_rope: int,
    v_head: int,
    q_lora: int,
    kv_lora: int,
    inter: int,
) -> KimiK3DecoderLayer:
    q_head_dim = qk_nope + qk_rope
    attention = KimiK3MLAAttention(
        hidden_size=hidden,
        num_heads=heads,
        qk_nope_head_dim=qk_nope,
        qk_rope_head_dim=qk_rope,
        v_head_dim=v_head,
        q_lora_rank=q_lora,
        kv_lora_rank=kv_lora,
        layer_idx=_LAYER_IDX,
    )

    def bf16(arr: np.ndarray) -> Tensor:
        return Tensor(arr, dtype=DType.float32).cast(DType.bfloat16)

    attention.q_a_proj.weight = bf16(_arr(q_lora, hidden))
    attention.q_a_layernorm.weight = bf16(np.ones(q_lora, dtype=np.float32))
    attention.q_b_proj.weight = bf16(_arr(heads * q_head_dim, q_lora))
    attention.kv_a_proj_with_mqa.weight = bf16(_arr(kv_lora + qk_rope, hidden))
    attention.kv_a_layernorm.weight = bf16(np.ones(kv_lora, dtype=np.float32))
    attention.kv_b_proj.weight = bf16(_arr(heads * (qk_nope + v_head), kv_lora))
    attention.g_proj.weight = bf16(_arr(heads * v_head, hidden))
    attention.o_proj.weight = bf16(_arr(hidden, heads * v_head))

    ffn = KimiK3DenseMLP(
        hidden_size=hidden, intermediate_size=inter, beta=1.5, linear_beta=1.5
    )
    ffn.gate_up_proj.weight = bf16(_arr(2 * inter, hidden))
    ffn.down_proj.weight = bf16(_arr(hidden, inter))

    layer = KimiK3DecoderLayer(
        layer_idx=_LAYER_IDX,
        attn_res_block_size=_ATTN_RES_BLOCK_SIZE,
        is_kda=False,
        is_dense=True,
        hidden_size=hidden,
        attention=attention,
        ffn=ffn,
    )
    layer.input_layernorm_weight = bf16(np.ones(hidden, dtype=np.float32))
    layer.post_attention_layernorm_weight = bf16(
        np.ones(hidden, dtype=np.float32)
    )
    layer.self_attention_res_proj_weight = bf16(_arr(1, hidden))
    layer.self_attention_res_norm_weight = bf16(
        np.ones(hidden, dtype=np.float32)
    )
    layer.mlp_res_proj_weight = bf16(_arr(1, hidden))
    layer.mlp_res_norm_weight = bf16(np.ones(hidden, dtype=np.float32))
    return layer


@pytest.mark.compile_on_gpu(
    "mo.mha.no_cache is staged as an undeclared custom op the interpreter can't bind"
)
def test_decoder_layer_mla_decompressed_ragged_matches_padded() -> None:
    hidden, heads, qk_nope, qk_rope, v_head, q_lora, kv_lora, inter = (
        64,
        4,
        16,
        8,
        16,
        32,
        24,
        96,
    )
    total_seq_len = 6
    x_np = _arr(total_seq_len, hidden)
    block_residual_np = [_arr(total_seq_len, hidden)]

    device = Accelerator()
    with default_device(device):
        kv_params = MHAKVCacheParams(
            dtype=DType.bfloat16,
            n_kv_heads=heads,
            head_dim=qk_nope + qk_rope,
            num_layers=1,
            devices=[DeviceRef.from_device(device)],
            page_size=_PAGE_SIZE,
        )
        layer = _build_layer(
            hidden=hidden,
            heads=heads,
            qk_nope=qk_nope,
            qk_rope=qk_rope,
            v_head=v_head,
            q_lora=q_lora,
            kv_lora=kv_lora,
            inter=inter,
        )
        assert isinstance(layer.attention, KimiK3MLAAttention)
        x = Tensor(x_np, dtype=DType.float32).cast(DType.bfloat16)
        block_residual = [
            Tensor(c, dtype=DType.float32).cast(DType.bfloat16)
            for c in block_residual_np
        ]
        out_padded, _ = layer(x, block_residual)
        out_padded_np = out_padded.cast(DType.float32).to_numpy()

        layer.attention.kv_params = kv_params
        kv_collection = build_paged_cache(kv_params, [total_seq_len], device)
        input_row_offsets = Tensor(
            np.array([0, total_seq_len], dtype=np.uint32), dtype=DType.uint32
        )
        out_ragged, _ = layer(
            x,
            block_residual,
            input_row_offsets=input_row_offsets,
            kv_collection=kv_collection,
        )
        out_ragged_np = out_ragged.cast(DType.float32).to_numpy()

    err = _rel_error(out_ragged_np, out_padded_np)
    assert err < 1e-2, f"decompressed ragged decoder layer mismatch: {err}"


@pytest.mark.compile_on_gpu(
    "mo.mha.no_cache is staged as an undeclared custom op the interpreter can't bind"
)
def test_decoder_layer_mla_latent_ragged_matches_padded() -> None:
    # The fused MLA prefill/decode kernel only instantiates for a 576-wide
    # latent cache (512 + 64) -- see `kimi_k3_modulev3/tests/test_mla_attention_latent_ragged.py`.
    hidden, heads, qk_nope, v_head, q_lora, inter = 256, 8, 64, 64, 128, 96
    kv_lora, qk_rope = 512, 64
    total_seq_len = 6
    x_np = _arr(total_seq_len, hidden)
    block_residual_np = [_arr(total_seq_len, hidden)]

    device = Accelerator()
    with default_device(device):
        kv_params = MLAKVCacheParams(
            dtype=DType.bfloat16,
            head_dim=kv_lora + qk_rope,
            num_layers=1,
            devices=[DeviceRef.from_device(device)],
            page_size=_PAGE_SIZE,
            num_q_heads=heads,
        )
        layer = _build_layer(
            hidden=hidden,
            heads=heads,
            qk_nope=qk_nope,
            qk_rope=qk_rope,
            v_head=v_head,
            q_lora=q_lora,
            kv_lora=kv_lora,
            inter=inter,
        )
        assert isinstance(layer.attention, KimiK3MLAAttention)
        x = Tensor(x_np, dtype=DType.float32).cast(DType.bfloat16)
        block_residual = [
            Tensor(c, dtype=DType.float32).cast(DType.bfloat16)
            for c in block_residual_np
        ]
        out_padded, _ = layer(x, block_residual)
        out_padded_np = out_padded.cast(DType.float32).to_numpy()

        layer.attention.kv_params = kv_params
        kv_collection = build_paged_cache(kv_params, [total_seq_len], device)
        input_row_offsets = Tensor(
            np.array([0, total_seq_len], dtype=np.uint32), dtype=DType.uint32
        )
        prefill_plan = kimi_k3_mla_prefill_plan(
            kv_params=kv_params,
            kv_collection=kv_collection,
            input_row_offsets=input_row_offsets,
            buffer_size=_PAGE_SIZE,
        )
        freqs_cis = kimi_k3_nope_freqs_cis(
            max_seq_len=total_seq_len,
            qk_rope_head_dim=qk_rope,
            dtype=DType.bfloat16,
            device=device,
        )
        out_ragged, _ = layer(
            x,
            block_residual,
            input_row_offsets=input_row_offsets,
            kv_collection=kv_collection,
            mla_prefill_plan=prefill_plan,
            mla_freqs_cis=freqs_cis,
        )
        out_ragged_np = out_ragged.cast(DType.float32).to_numpy()

    err = _rel_error(out_ragged_np, out_padded_np)
    assert err < 1e-2, f"latent ragged decoder layer mismatch: {err}"
