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
"""Verifies the latent ragged MLA path matches the padded parity form.

Builds one layer with real weights and runs both ``forward`` (padded,
already proven against the reference checkpoint) and
``forward_latent_ragged`` (a single prefill into an empty cache) over the
same input, then compares outputs -- the only numerical difference between
the two is which kernel computes the same attention, so they should match
closely.
"""

from __future__ import annotations

import numpy as np
import pytest
from _paged_cache_test_utils import build_paged_cache
from max.driver import Accelerator
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.legacy.kv_cache import MLAKVCacheParams
from max.pipelines.architectures.kimi_k3_modulev3.layers.mla_attention import (
    KimiK3MLAAttention,
    kimi_k3_mla_prefill_plan,
    kimi_k3_nope_freqs_cis,
)
from max.tensor import Tensor, default_device

_HIDDEN = 256
_NUM_HEADS = 8
_QK_NOPE_HEAD_DIM = 64
_V_HEAD_DIM = 64
_Q_LORA_RANK = 128
_SEQ_LEN = 6
_PAGE_SIZE = 128

# The fused MLA prefill/decode kernel only instantiates for a 576-wide latent
# cache (512 + 64) -- `flareMLA_decoding`'s `constraint failed: ... head_dim
# 576 (rotary) or 512 (NoPE)`. Real K3 and K3-mini are exactly this shape; see
# `kimi_k3/model_config.py`'s `KimiK3Config._MLA_KERNEL_LATENT_DIM`.
_KV_LORA_RANK = 512
_QK_ROPE_HEAD_DIM = 64


@pytest.mark.compile_on_gpu(
    "mo.mha.no_cache is staged as an undeclared custom op the interpreter can't bind"
)
def test_forward_latent_ragged_matches_forward() -> None:
    rng = np.random.default_rng(0)

    def arr(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.02).astype(np.float32)

    device = Accelerator()
    with default_device(device):
        layer = KimiK3MLAAttention(
            hidden_size=_HIDDEN,
            num_heads=_NUM_HEADS,
            qk_nope_head_dim=_QK_NOPE_HEAD_DIM,
            qk_rope_head_dim=_QK_ROPE_HEAD_DIM,
            v_head_dim=_V_HEAD_DIM,
            q_lora_rank=_Q_LORA_RANK,
            kv_lora_rank=_KV_LORA_RANK,
            kv_params=MLAKVCacheParams(
                dtype=DType.bfloat16,
                head_dim=_KV_LORA_RANK + _QK_ROPE_HEAD_DIM,
                num_layers=1,
                devices=[DeviceRef.from_device(device)],
                page_size=_PAGE_SIZE,
                num_q_heads=_NUM_HEADS,
            ),
            layer_idx=0,
        )
        layer.q_a_proj.weight = Tensor(
            arr(_Q_LORA_RANK, _HIDDEN), dtype=DType.float32
        ).cast(DType.bfloat16)
        layer.q_a_layernorm.weight = Tensor(
            np.ones(_Q_LORA_RANK, dtype=np.float32), dtype=DType.float32
        ).cast(DType.bfloat16)
        layer.q_b_proj.weight = Tensor(
            arr(
                _NUM_HEADS * (_QK_NOPE_HEAD_DIM + _QK_ROPE_HEAD_DIM),
                _Q_LORA_RANK,
            ),
            dtype=DType.float32,
        ).cast(DType.bfloat16)
        layer.kv_a_proj_with_mqa.weight = Tensor(
            arr(_KV_LORA_RANK + _QK_ROPE_HEAD_DIM, _HIDDEN), dtype=DType.float32
        ).cast(DType.bfloat16)
        layer.kv_a_layernorm.weight = Tensor(
            np.ones(_KV_LORA_RANK, dtype=np.float32), dtype=DType.float32
        ).cast(DType.bfloat16)
        layer.kv_b_proj.weight = Tensor(
            arr(_NUM_HEADS * (_QK_NOPE_HEAD_DIM + _V_HEAD_DIM), _KV_LORA_RANK),
            dtype=DType.float32,
        ).cast(DType.bfloat16)
        layer.g_proj.weight = Tensor(
            arr(_NUM_HEADS * _V_HEAD_DIM, _HIDDEN), dtype=DType.float32
        ).cast(DType.bfloat16)
        layer.o_proj.weight = Tensor(
            arr(_HIDDEN, _NUM_HEADS * _V_HEAD_DIM), dtype=DType.float32
        ).cast(DType.bfloat16)

        x_np = arr(_SEQ_LEN, _HIDDEN)
        x = Tensor(x_np, dtype=DType.float32).cast(DType.bfloat16)

        padded = Tensor(
            x_np.reshape(1, _SEQ_LEN, _HIDDEN), dtype=DType.float32
        ).cast(DType.bfloat16)
        out_padded = layer(padded).cast(DType.float32).to_numpy()[0]

        assert isinstance(layer.kv_params, MLAKVCacheParams)
        kv_collection = build_paged_cache(layer.kv_params, [_SEQ_LEN], device)
        input_row_offsets = Tensor(
            np.array([0, _SEQ_LEN], dtype=np.uint32), dtype=DType.uint32
        )
        prefill_plan = kimi_k3_mla_prefill_plan(
            kv_params=layer.kv_params,
            kv_collection=kv_collection,
            input_row_offsets=input_row_offsets,
            buffer_size=_PAGE_SIZE,
        )
        freqs_cis = kimi_k3_nope_freqs_cis(
            max_seq_len=_SEQ_LEN,
            qk_rope_head_dim=_QK_ROPE_HEAD_DIM,
            dtype=DType.bfloat16,
            device=device,
        )
        out_latent_ragged = (
            layer.forward_latent_ragged(
                x, kv_collection, input_row_offsets, prefill_plan, freqs_cis
            )
            .cast(DType.float32)
            .to_numpy()
        )

    err = float(
        np.max(np.abs(out_latent_ragged - out_padded))
        / max(np.max(np.abs(out_padded)), 1e-12)
    )
    assert err < 1e-2, f"latent ragged MLA vs padded parity diverges: {err}"
