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
"""Verifies two KDA layers reading a shared, multi-layer state pool don't
cross-contaminate each other's rows.

Runs each of two KDA decoder layers once against its own private, single-row
pool (so a row collision is structurally impossible) and once against a
single two-row pool with distinct row ids, in the execution order
``KimiK3Model`` uses (layer 0 then layer 1, so layer 1 sees whatever layer 0
already wrote). The two must match; if the row ids instead collided, layer
1's shared-pool run would read layer 0's just-written state instead of a
clean initial state, and diverge from its private-pool run.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import Accelerator
from max.dtype import DType
from max.pipelines.architectures.kimi_k3_modulev3.layers.decoder_layer import (
    KimiK3DecoderLayer,
)
from max.pipelines.architectures.kimi_k3_modulev3.layers.kda_attention import (
    KimiK3KDAAttention,
)
from max.pipelines.architectures.kimi_k3_modulev3.layers.mlp import (
    KimiK3DenseMLP,
)
from max.tensor import Tensor, default_device

_RNG = np.random.default_rng(20261022)

_HIDDEN = 16
_HEADS = 2
_HEAD_DIM = 32  # kda_decode only supports compiled head_dim shapes 32 or 128.
_CONV_KERNEL = 4
_INTER = 24
_TOTAL_SEQ_LEN = 4


def _arr(*shape: int) -> np.ndarray:
    return (_RNG.standard_normal(shape) * 0.02).astype(np.float32)


def _build_layer() -> KimiK3DecoderLayer:
    inner = _HEADS * _HEAD_DIM
    attention = KimiK3KDAAttention(
        hidden_size=_HIDDEN,
        num_heads=_HEADS,
        head_dim=_HEAD_DIM,
        conv_kernel_size=_CONV_KERNEL,
        gate_lower_bound=None,
    )
    attention.q_proj.weight = Tensor(_arr(inner, _HIDDEN), dtype=DType.float32)
    attention.k_proj.weight = Tensor(_arr(inner, _HIDDEN), dtype=DType.float32)
    attention.v_proj.weight = Tensor(_arr(inner, _HIDDEN), dtype=DType.float32)
    attention.f_a_proj.weight = Tensor(
        _arr(_HEAD_DIM, _HIDDEN), dtype=DType.float32
    )
    attention.f_b_proj.weight = Tensor(
        _arr(inner, _HEAD_DIM), dtype=DType.float32
    )
    attention.b_proj.weight = Tensor(_arr(_HEADS, _HIDDEN), dtype=DType.float32)
    attention.g_proj.weight = Tensor(_arr(inner, _HIDDEN), dtype=DType.float32)
    attention.o_proj.weight = Tensor(_arr(_HIDDEN, inner), dtype=DType.float32)
    attention.o_norm.weight = Tensor(
        (1.0 + 0.1 * _RNG.standard_normal(_HEAD_DIM)).astype(np.float32),
        dtype=DType.float32,
    )
    attention.conv_weight = Tensor(
        _arr(3 * inner, _CONV_KERNEL), dtype=DType.float32
    )
    attention.a_log = Tensor(
        np.linspace(-1.0, 0.5, _HEADS * _HEAD_DIM, dtype=np.float32).reshape(
            _HEADS, _HEAD_DIM
        ),
        dtype=DType.float32,
    )
    attention.dt_bias = Tensor(_arr(_HEADS, _HEAD_DIM), dtype=DType.float32)

    ffn = KimiK3DenseMLP(
        hidden_size=_HIDDEN, intermediate_size=_INTER, beta=1.5, linear_beta=1.5
    )
    ffn.gate_up_proj.weight = Tensor(
        _arr(2 * _INTER, _HIDDEN), dtype=DType.float32
    )
    ffn.down_proj.weight = Tensor(_arr(_HIDDEN, _INTER), dtype=DType.float32)

    layer = KimiK3DecoderLayer(
        layer_idx=0,
        attn_res_block_size=1,
        is_kda=True,
        is_dense=True,
        hidden_size=_HIDDEN,
        attention=attention,
        ffn=ffn,
    )
    layer.input_layernorm_weight = Tensor(
        np.ones(_HIDDEN, dtype=np.float32), dtype=DType.float32
    )
    layer.post_attention_layernorm_weight = Tensor(
        np.ones(_HIDDEN, dtype=np.float32), dtype=DType.float32
    )
    layer.self_attention_res_proj_weight = Tensor(
        _arr(1, _HIDDEN), dtype=DType.float32
    )
    layer.self_attention_res_norm_weight = Tensor(
        np.ones(_HIDDEN, dtype=np.float32), dtype=DType.float32
    )
    layer.mlp_res_proj_weight = Tensor(_arr(1, _HIDDEN), dtype=DType.float32)
    layer.mlp_res_norm_weight = Tensor(
        np.ones(_HIDDEN, dtype=np.float32), dtype=DType.float32
    )
    return layer


@pytest.mark.compile_on_gpu("gated_delta_conv1d_fwd mutates its state in place")
def test_shared_pool_rows_do_not_collide() -> None:
    x_np = _arr(_TOTAL_SEQ_LEN, _HIDDEN)
    row_offsets = Tensor(
        np.array([0, _TOTAL_SEQ_LEN], dtype=np.uint32), dtype=DType.uint32
    )
    inner = _HEADS * _HEAD_DIM
    conv_shape_one = [1, 3 * inner, _CONV_KERNEL - 1]
    state_shape_one = [1, _HEADS, _HEAD_DIM, _HEAD_DIM]

    with default_device(Accelerator()):
        layer0 = _build_layer()
        layer1 = _build_layer()
        row0 = Tensor(np.array([0], dtype=np.uint32), dtype=DType.uint32)

        # Private pools: a collision is structurally impossible here.
        out0_ref, _ = layer0(
            Tensor(x_np, dtype=DType.float32),
            [],
            conv_pool=Tensor.zeros(conv_shape_one, dtype=DType.float32),
            conv_row_id=row0,
            state_pool=Tensor.zeros(state_shape_one, dtype=DType.float32),
            recurrent_row_id=row0,
            input_row_offsets=row_offsets,
        )
        out1_ref, _ = layer1(
            Tensor(x_np, dtype=DType.float32),
            [],
            conv_pool=Tensor.zeros(conv_shape_one, dtype=DType.float32),
            conv_row_id=row0,
            state_pool=Tensor.zeros(state_shape_one, dtype=DType.float32),
            recurrent_row_id=row0,
            input_row_offsets=row_offsets,
        )

        # One shared, two-row pool, with distinct rows -- layer 0 runs (and
        # writes row 0) before layer 1 reads row 1.
        shared_conv = Tensor.zeros(
            [2, 3 * inner, _CONV_KERNEL - 1], dtype=DType.float32
        )
        shared_state = Tensor.zeros(
            [2, _HEADS, _HEAD_DIM, _HEAD_DIM], dtype=DType.float32
        )
        row1 = Tensor(np.array([1], dtype=np.uint32), dtype=DType.uint32)
        out0_shared, _ = layer0(
            Tensor(x_np, dtype=DType.float32),
            [],
            conv_pool=shared_conv,
            conv_row_id=row0,
            state_pool=shared_state,
            recurrent_row_id=row0,
            input_row_offsets=row_offsets,
        )
        out1_shared, _ = layer1(
            Tensor(x_np, dtype=DType.float32),
            [],
            conv_pool=shared_conv,
            conv_row_id=row1,
            state_pool=shared_state,
            recurrent_row_id=row1,
            input_row_offsets=row_offsets,
        )

        out0_ref_np = out0_ref.to_numpy()
        out1_ref_np = out1_ref.to_numpy()
        out0_shared_np = out0_shared.to_numpy()
        out1_shared_np = out1_shared.to_numpy()

    assert np.allclose(out0_shared_np, out0_ref_np, atol=1e-5), (
        "layer 0 should be unaffected by sharing the pool"
    )
    assert np.allclose(out1_shared_np, out1_ref_np, atol=1e-5), (
        "layer 1's shared-pool row collided with layer 0's"
    )
