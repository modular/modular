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

"""Graph-level A/B of the AMD MoE FFN op (`mo.grouped.ffn.block.scaled.amd`)
at MXFP4, MiniMax-M3's expert shape, against the three-op chain it replaces:
gate/up into bf16 (W1 split `[gate | up]`), `ep.fused_silu.mxfp4`, down,
byte for byte.

In the decode band the op runs the chain for the smallest steps and the
one-launch MegaFFN above them, on one workgroup a CU for small steps and two
above, each on its own half of a persistent counter
buffer zeroed once at load; the graph executes several times so a stale
count across executions would show. Above its prefill band it runs the
prefill MegaFFN (or the two prefill GEMMs), here on partial 128-row tiles
and an empty expert. Elsewhere the op runs the chain itself.

NOTE: AMD gfx950 only.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import Accelerator, Buffer, accelerator_api
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, Graph, TensorType, ops
from max.nn.legacy.activation import GatedActivation
from max.nn.legacy.kernels import (
    amd_mega_ffn_workspace_words,
    block_scaled_preshuffle_grouped_scale_4d,
    grouped_dynamic_block_scaled_matmul_amd,
    grouped_ffn_block_scaled_amd,
    preshuffle_block_scaled_b,
    preshuffle_block_scaled_b_scales,
)

H = 6144
I = 3072
N1 = 2 * I
SLOT_ROWS = 32


def _fp4_bytes(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    """Uniform bytes: every pair of E2M1 nibbles is finite."""
    return rng.integers(0, 256, size=shape, dtype=np.uint8)


def _scale_bytes(
    rng: np.random.Generator, shape: tuple[int, ...], lo: int
) -> np.ndarray:
    return (127 + lo + rng.integers(0, 3, size=shape)).astype(np.uint8)


@pytest.mark.skipif(
    accelerator_api() != "hip", reason="the AMD MoE FFN is gfx950-only"
)
@pytest.mark.parametrize(
    "regime",
    ["gated", "mega_small", "mega", "prefill_mega", "prefill_gemm", "chain"],
)
def test_mega_ffn_amd_mxfp4_matches_chain(regime: str) -> None:
    # "mega_small" stays under the op's one-workgroup-a-CU decode bound;
    # "gated" under its MXFP4 chain bound, inside the decode band
    if regime == "gated":
        tokens_per_expert = [1, 0, 1, 0]
    elif regime == "mega_small":
        tokens_per_expert = [2, 0, 1, 1]
    else:
        tokens_per_expert = [16, 0, 21, 9]
    n_experts = len(tokens_per_expert)
    total = sum(tokens_per_expert)
    max_tok = max(tokens_per_expert)
    assert max_tok <= SLOT_ROWS
    rng = np.random.default_rng(11)
    x = _fp4_bytes(rng, (total, H // 2))
    sx = _scale_bytes(rng, (total, H // 32), -4)
    w1 = _fp4_bytes(rng, (n_experts, N1, H // 2))
    s1 = _scale_bytes(rng, (n_experts, N1, H // 32), -4)
    w2 = _fp4_bytes(rng, (n_experts, H, I // 2))
    s2 = _scale_bytes(rng, (n_experts, H, I // 32), -4)
    offsets = np.zeros(n_experts + 1, dtype=np.uint32)
    offsets[1:] = np.cumsum(tokens_per_expert)
    expert_ids = np.arange(n_experts, dtype=np.int32)
    usage = np.array([max_tok, n_experts], dtype=np.uint32)

    device = Accelerator()
    device_ref = DeviceRef(device.label, device.id)
    cpu_ref = DeviceRef.CPU()
    session = InferenceSession(devices=[device])
    e8 = DType.float8_e8m0fnu
    u8 = DType.uint8
    input_types = [
        TensorType(u8, (total, H // 2), device=device_ref),
        TensorType(e8, (total, H // 32), device=device_ref),
        TensorType(u8, (n_experts, N1, H // 2), device=device_ref),
        TensorType(e8, (n_experts, N1, H // 32), device=device_ref),
        TensorType(u8, (n_experts, H, I // 2), device=device_ref),
        TensorType(e8, (n_experts, H, I // 32), device=device_ref),
        TensorType(DType.uint32, (n_experts + 1,), device=device_ref),
        TensorType(DType.int32, (n_experts,), device=device_ref),
        TensorType(DType.uint32, (2,), device=cpu_ref),
    ]
    decode_band = regime.startswith("mega") or regime == "gated"
    with Graph("mega_ffn_amd_mxfp4_ab", input_types=input_types) as graph:
        (x_t, sx_t, w1_t, s1_t, w2_t, s2_t, off_t, ids_t, usage_t) = (
            v.tensor for v in graph.inputs
        )
        # A-scales as EP dispatch writes them: the per-expert slot layout
        sx_slot = block_scaled_preshuffle_grouped_scale_4d(
            sx_t,
            off_t,
            usage_t[0].cast(DType.uint32),
            usage_t[1].cast(DType.uint32),
            num_experts=n_experts,
        )
        w1_pre = preshuffle_block_scaled_b(w1_t)
        w2_pre = preshuffle_block_scaled_b(w2_t)
        s1_pre = preshuffle_block_scaled_b_scales(s1_t)
        s2_pre = preshuffle_block_scaled_b_scales(s2_t)
        est_m = ops.constant(total, dtype=DType.uint32, device=cpu_ref)

        gate_up = grouped_dynamic_block_scaled_matmul_amd(
            x_t,
            w1_pre,
            sx_slot,
            s1_pre,
            off_t,
            ids_t,
            usage_t,
            estimated_total_m=est_m,
            preshuffled_b=True,
            a_scales_max_padded_m=SLOT_ROWS,
        )
        q, q_scales = (
            r.tensor
            for r in ops.custom(
                "ep.fused_silu.mxfp4",
                device=device_ref,
                values=[gate_up, off_t],
                out_types=[
                    TensorType(u8, (total, I // 2), device=device_ref),
                    TensorType(
                        e8, (n_experts * SLOT_ROWS, I // 32), device=device_ref
                    ),
                ],
                parameters={
                    **GatedActivation().kernel_parameters(),
                    "max_padded_M": SLOT_ROWS,
                },
            )
        )
        y_ref = grouped_dynamic_block_scaled_matmul_amd(
            q,
            w2_pre,
            q_scales,
            s2_pre,
            off_t,
            ids_t,
            usage_t,
            estimated_total_m=est_m,
            preshuffled_b=True,
            a_scales_max_padded_m=SLOT_ROWS,
        )
        # "gated": decode regions of one slot each, which the MegaFFN
        # refuses, so the step must take the chain to pass at all.
        words = amd_mega_ffn_workspace_words(n_experts, total, total, grids=2)
        if regime == "gated":
            decode = max(n_experts + -(-total // 16), 1024)
            region = words - amd_mega_ffn_workspace_words(
                n_experts, total, total
            )
            slot = (region - 2) // decode
            words -= 2 * (decode - 1) * slot
        workspace = ops.buffer_create(
            BufferType(DType.uint32, shape=[words], device=device_ref),
            init_value=0,
        )
        y = grouped_ffn_block_scaled_amd(
            x_t,
            w1_pre,
            sx_slot,
            s1_pre,
            w2_pre,
            s2_pre,
            off_t,
            ids_t,
            usage_t,
            workspace,
            est_m,
            scales_max_padded_m=SLOT_ROWS,
            decode_grid_m_cap=0,
            decode_grid_m_rows=max_tok,
            mega_band_m=256 if decode_band else 0,
            mega_max_rows=total,
            prefill_band_m=1 if regime.startswith("prefill") else 0,
            prefill_mega=regime == "prefill_mega",
        )
        # bf16 -> f32 is exact, so equal f32 means equal bf16 bits
        graph.output(y_ref.cast(DType.float32), y.cast(DType.float32))

    compiled = session.load(graph)
    cpu = Accelerator.cpu()

    def dev(a: np.ndarray, dtype: DType | None = None) -> Buffer:
        b = Buffer.from_numpy(a)
        if dtype is not None:
            b = b.view(dtype)
        return b.to(device)

    args = [
        dev(x),
        dev(sx, e8),
        dev(w1),
        dev(s1, e8),
        dev(w2),
        dev(s2, e8),
        dev(offsets),
        dev(expert_ids),
        Buffer.from_numpy(usage),
    ]
    for execution in range(3):
        y_ref_buf, y_buf = compiled.execute(*args)
        y_ref_np = np.from_dlpack(y_ref_buf.to(cpu))
        y_np = np.from_dlpack(y_buf.to(cpu))
        assert np.count_nonzero(y_ref_np) > y_ref_np.size // 2
        bad = int(np.count_nonzero(y_ref_np != y_np))
        assert bad == 0, (
            f"execution {execution}: {bad} of {y_np.size} bf16 outputs"
            f" differ from the chain (regime={regime})"
        )
