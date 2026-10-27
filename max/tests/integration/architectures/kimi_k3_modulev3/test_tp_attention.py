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
"""TP=1 vs TP=2 numerical parity for KimiK3MLAAttention and KimiK3KDAAttention.

Builds each layer at TP=1 and TP=2 with identical weights and checks that
the allreduced TP=2 output matches TP=1.
"""

from __future__ import annotations

import os

os.environ.setdefault("MODULAR_DEBUG", "source-tracebacks")

import numpy as np
import pytest
from max import functional as F
from max.driver import Accelerator
from max.dtype import DType
from max.pipelines.architectures.kimi_k3_modulev3.layers.kda_attention import (
    KimiK3KDAAttention,
    tensor_parallel_kda_attention,
)
from max.pipelines.architectures.kimi_k3_modulev3.layers.mla_attention import (
    KimiK3MLAAttention,
    tensor_parallel_mla_attention,
)
from max.tensor import Tensor, default_device
from max.tensor.sharding import DeviceMesh, NamedMapping

_RNG = np.random.default_rng(20261006)


def _rel_error(got: np.ndarray, want: np.ndarray) -> float:
    return float(np.max(np.abs(got - want)) / max(np.max(np.abs(want)), 1e-12))


@pytest.mark.compile_on_gpu(
    "mo.mha.no_cache is staged as an undeclared custom op the interpreter can't bind"
)
def test_mla_tp1_matches_tp2() -> None:
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

    def make(num_heads: int) -> dict[str, np.ndarray]:
        return {
            "q_a_proj": _RNG.standard_normal((q_lora, hidden)).astype(
                np.float32
            )
            / np.sqrt(np.float32(hidden)),
            "q_b_proj": _RNG.standard_normal(
                (num_heads * q_head_dim, q_lora)
            ).astype(np.float32)
            / np.sqrt(np.float32(q_lora)),
            "kv_a_proj_with_mqa": _RNG.standard_normal(
                (kv_lora + qk_rope, hidden)
            ).astype(np.float32)
            / np.sqrt(np.float32(hidden)),
            "kv_b_proj": _RNG.standard_normal(
                (num_heads * (qk_nope + v_head), kv_lora)
            ).astype(np.float32)
            / np.sqrt(np.float32(kv_lora)),
            "g_proj": _RNG.standard_normal((num_heads * v_head, hidden)).astype(
                np.float32
            )
            / np.sqrt(np.float32(hidden)),
            "o_proj": _RNG.standard_normal((hidden, num_heads * v_head)).astype(
                np.float32
            )
            / np.sqrt(np.float32(num_heads * v_head)),
        }

    weights = make(heads)
    x_np = _RNG.standard_normal((2, 5, hidden)).astype(np.float32)

    # TP=1.
    with default_device(Accelerator(0)):
        layer1 = KimiK3MLAAttention(
            hidden_size=hidden,
            num_heads=heads,
            qk_nope_head_dim=qk_nope,
            qk_rope_head_dim=qk_rope,
            v_head_dim=v_head,
            q_lora_rank=q_lora,
            kv_lora_rank=kv_lora,
        )
        for name, arr in weights.items():
            w = getattr(layer1, name)
            w.weight = Tensor(arr, dtype=DType.float32).cast(w.weight.dtype)
        x1 = Tensor(x_np, dtype=DType.float32).cast(DType.bfloat16)
        out1 = layer1(x1).cast(DType.float32).to_numpy()

    # TP=2.
    devices = [Accelerator(0), Accelerator(1)]
    mesh = DeviceMesh(
        devices=tuple(devices), mesh_shape=(2,), axis_names=("tp",)
    )
    with default_device(mesh):
        layer2 = KimiK3MLAAttention(
            hidden_size=hidden,
            num_heads=heads,
            qk_nope_head_dim=qk_nope,
            qk_rope_head_dim=qk_rope,
            v_head_dim=v_head,
            q_lora_rank=q_lora,
            kv_lora_rank=kv_lora,
        )
        for name, arr in weights.items():
            w = getattr(layer2, name)
            w.weight = Tensor(arr, dtype=DType.float32).cast(w.weight.dtype)
        tensor_parallel_mla_attention(layer2, mesh)
        x2 = Tensor(x_np, dtype=DType.float32).cast(DType.bfloat16)
        partial = layer2(x2)
        full = F.allreduce_sum(partial)
        out2 = full.cast(DType.float32).to_numpy()

    rel_error = _rel_error(out2, out1)
    print(f"MLA TP=1 vs TP=2 rel_error={rel_error}")
    assert rel_error < 1e-2, f"MLA TP mismatch: {rel_error}"


@pytest.mark.compile_on_gpu("gated_delta_conv1d_fwd mutates its state in place")
@pytest.mark.parametrize("gate_lower_bound", [None, -5.0])
def test_kda_tp1_matches_tp2(gate_lower_bound: float | None) -> None:
    hidden, heads, head_dim, conv_kernel = 64, 8, 32, 4
    inner = heads * head_dim

    def make() -> dict[str, np.ndarray]:
        return {
            "q_proj": _RNG.standard_normal((inner, hidden)).astype(np.float32)
            / np.sqrt(np.float32(hidden)),
            "k_proj": _RNG.standard_normal((inner, hidden)).astype(np.float32)
            / np.sqrt(np.float32(hidden)),
            "v_proj": _RNG.standard_normal((inner, hidden)).astype(np.float32)
            / np.sqrt(np.float32(hidden)),
            "f_a_proj": _RNG.standard_normal((head_dim, hidden)).astype(
                np.float32
            )
            / np.sqrt(np.float32(hidden)),
            "f_b_proj": _RNG.standard_normal((inner, head_dim)).astype(
                np.float32
            )
            / np.sqrt(np.float32(head_dim)),
            "b_proj": _RNG.standard_normal((heads, hidden)).astype(np.float32)
            / np.sqrt(np.float32(hidden)),
            "g_proj": _RNG.standard_normal((inner, hidden)).astype(np.float32)
            / np.sqrt(np.float32(hidden)),
            "o_proj": _RNG.standard_normal((hidden, inner)).astype(np.float32)
            / np.sqrt(np.float32(inner)),
        }

    weights = make()
    conv_weight_np = _RNG.standard_normal((3 * inner, conv_kernel)).astype(
        np.float32
    ) / np.sqrt(np.float32(conv_kernel))
    a_log_np = np.linspace(
        -1.0, 0.5, heads * head_dim, dtype=np.float32
    ).reshape(heads, head_dim)
    dt_bias_np = _RNG.standard_normal((heads, head_dim)).astype(np.float32)
    o_norm_np = (
        1.0 + 0.1 * _RNG.standard_normal(head_dim).astype(np.float32)
    ).astype(np.float32)

    total_seq_len, batch_size, max_slots = 6, 2, 2
    # Two steps on the same pools, so the second reads the state the first
    # wrote.
    steps_np = [
        _RNG.standard_normal((total_seq_len, hidden)).astype(np.float32)
        for _ in range(2)
    ]
    input_row_offsets_np = np.array(
        [i * (total_seq_len // batch_size) for i in range(batch_size + 1)],
        dtype=np.uint32,
    )
    slot_idx_np = np.arange(batch_size, dtype=np.uint32)

    def assign(layer: KimiK3KDAAttention) -> None:
        layer.q_proj.weight = Tensor(weights["q_proj"], dtype=DType.float32)
        layer.k_proj.weight = Tensor(weights["k_proj"], dtype=DType.float32)
        layer.v_proj.weight = Tensor(weights["v_proj"], dtype=DType.float32)
        layer.f_a_proj.weight = Tensor(weights["f_a_proj"], dtype=DType.float32)
        layer.f_b_proj.weight = Tensor(weights["f_b_proj"], dtype=DType.float32)
        layer.b_proj.weight = Tensor(weights["b_proj"], dtype=DType.float32)
        layer.g_proj.weight = Tensor(weights["g_proj"], dtype=DType.float32)
        layer.o_proj.weight = Tensor(weights["o_proj"], dtype=DType.float32)
        layer.conv_weight = Tensor(conv_weight_np, dtype=DType.float32)
        layer.a_log = Tensor(a_log_np, dtype=DType.float32)
        layer.dt_bias = Tensor(dt_bias_np, dtype=DType.float32)
        layer.o_norm.weight = Tensor(o_norm_np, dtype=DType.float32)

    def run(
        layer: KimiK3KDAAttention, conv_pool: Tensor, state_pool: Tensor
    ) -> list[np.ndarray]:
        outs = []
        for x_np in steps_np:
            out = layer(
                Tensor(x_np, dtype=DType.float32),
                conv_pool,
                Tensor(slot_idx_np, dtype=DType.uint32),
                state_pool,
                Tensor(slot_idx_np, dtype=DType.uint32),
                Tensor(input_row_offsets_np, dtype=DType.uint32),
            )
            if out.is_distributed:
                out = F.allreduce_sum(out)
            outs.append(out.to_numpy())
        return outs

    def build() -> KimiK3KDAAttention:
        layer = KimiK3KDAAttention(
            hidden_size=hidden,
            num_heads=heads,
            head_dim=head_dim,
            conv_kernel_size=conv_kernel,
            gate_lower_bound=gate_lower_bound,
        )
        assign(layer)
        return layer

    conv_pool_shape = [max_slots, 3 * inner, conv_kernel - 1]
    state_pool_shape = [max_slots, heads, head_dim, head_dim]

    # TP=1.
    with default_device(Accelerator(0)):
        outs1 = run(
            build(),
            Tensor.zeros(conv_pool_shape, dtype=DType.float32),
            Tensor.zeros(state_pool_shape, dtype=DType.float32),
        )

    # TP=2.
    devices = [Accelerator(0), Accelerator(1)]
    mesh = DeviceMesh(
        devices=tuple(devices), mesh_shape=(2,), axis_names=("tp",)
    )
    with default_device(mesh):
        pool_mapping = NamedMapping(mesh, (None, "tp"))
        outs2 = run(
            tensor_parallel_kda_attention(build(), mesh),
            Tensor.zeros(
                conv_pool_shape, dtype=DType.float32, device=pool_mapping
            ),
            Tensor.zeros(
                state_pool_shape, dtype=DType.float32, device=pool_mapping
            ),
        )

    for step, (out1, out2) in enumerate(zip(outs1, outs2, strict=True)):
        rel_error = _rel_error(out2, out1)
        print(f"KDA step {step} TP=1 vs TP=2 rel_error={rel_error}")
        assert rel_error < 1e-2, f"KDA step {step} TP mismatch: {rel_error}"
