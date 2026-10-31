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
"""Verifies the decompressed ragged MLA path matches the padded parity form.

Builds one layer with real weights and runs both ``forward`` (padded,
already proven against the reference checkpoint) and ``forward_ragged``
(a single prefill into an empty cache) over the same input, then compares
outputs -- the only numerical difference between the two is which kernel
computes the same attention, so they should match closely.
"""

from __future__ import annotations

import numpy as np
import pytest
from _paged_cache_test_utils import build_paged_cache
from max.driver import Accelerator
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.legacy.kv_cache import MHAKVCacheParams
from max.pipelines.architectures.kimi_k3_modulev3.layers.mla_attention import (
    KimiK3MLAAttention,
)
from max.tensor import Tensor, default_device

_HIDDEN = 64
_NUM_HEADS = 4
_QK_NOPE_HEAD_DIM = 16
_QK_ROPE_HEAD_DIM = 8
_V_HEAD_DIM = 16
_Q_LORA_RANK = 32
_KV_LORA_RANK = 24
_SEQ_LEN = 6
_PAGE_SIZE = 128


@pytest.mark.compile_on_gpu(
    "mo.mha.no_cache is staged as an undeclared custom op the interpreter can't bind"
)
def test_forward_ragged_matches_forward() -> None:
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
            kv_params=MHAKVCacheParams(
                dtype=DType.bfloat16,
                n_kv_heads=_NUM_HEADS,
                head_dim=_QK_NOPE_HEAD_DIM + _QK_ROPE_HEAD_DIM,
                num_layers=1,
                devices=[DeviceRef.from_device(device)],
                page_size=_PAGE_SIZE,
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

        assert isinstance(layer.kv_params, MHAKVCacheParams)
        kv_collection = build_paged_cache(layer.kv_params, [_SEQ_LEN], device)
        input_row_offsets = Tensor(
            np.array([0, _SEQ_LEN], dtype=np.uint32), dtype=DType.uint32
        )
        out_ragged = (
            layer.forward_ragged(x, kv_collection, input_row_offsets)
            .cast(DType.float32)
            .to_numpy()
        )

    err = float(
        np.max(np.abs(out_ragged - out_padded))
        / max(np.max(np.abs(out_padded)), 1e-12)
    )
    assert err < 1e-2, f"ragged MLA vs padded parity diverges: {err}"
