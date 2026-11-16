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

"""Graph-level A/B of the AMD MoE FFN op at A8W4 under tensor parallelism,
and of the op that lays out its weights.

Kimi K3's expert shape on one of 8 GPUs (latent 3584, expert width 384, 896
experts, top 16), the op reading preshuffled weights, against the graph ops
K3 runs without it on the row-major ones: the dense W4A8 GEMM into bf16, the
fused SiTU MXFP8 quantize, the dense GEMM again. Byte for byte.

In its band the op runs the one-launch decode MegaFFN (a workspace too small
for it must raise, so the band is shown to take it); above, the prefill
MegaFFN. The graph executes several times on a persistent workspace zeroed
once at load.

NOTE: AMD gfx950 only.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import CPU, Accelerator, Buffer, accelerator_api
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, Graph, TensorType, ops
from max.nn._graph_ops import (
    amd_mega_ffn_workspace_words,
    grouped_dynamic_block_scaled_matmul_amd,
    grouped_ffn_block_scaled_amd,
    preshuffle_block_scaled_b,
    stack_a8w4_mega_ffn_b,
    swiglu_quantize_dynamic_block_scaled,
)
from max.nn.activation import GatedActivation

H = 3584
I = 384
N1 = 2 * I
E = 896
TOP_K = 16
BETA = 4.0
LINEAR_BETA = 25.0
BAND = 64 * TOP_K
SITU = GatedActivation(fused_activation="situ", alpha=BETA, limit=LINEAR_BETA)


def _rows(rng: np.random.Generator, tokens: int) -> np.ndarray:
    rows = np.zeros(E, dtype=np.int64)
    for _ in range(tokens):
        rows[rng.choice(E, TOP_K, replace=False)] += 1
    return rows


@pytest.mark.skipif(
    accelerator_api() != "hip", reason="the AMD MoE FFN is gfx950-only"
)
@pytest.mark.parametrize(
    "regime, tokens",
    [("mega", 1), ("mega", 9), ("mega", 64), ("prefill", 64), ("trap", 9)],
)
def test_mega_ffn_amd_a8w4_matches_graph_ops(regime: str, tokens: int) -> None:
    rng = np.random.default_rng(tokens)
    rows = _rows(rng, tokens)
    total = int(rows.sum())
    fp8 = rng.integers(0, 256, size=(total, H), dtype=np.uint8)
    fp8[(fp8 & 0x7F) == 0x7F] ^= 1
    sx = (127 - 9 + rng.integers(0, 3, size=(total, H // 32))).astype(np.uint8)
    w1 = rng.integers(0, 256, size=(E, N1, H // 2), dtype=np.uint8)
    s1 = (127 - 7 + rng.integers(0, 3, size=(E, N1, H // 32))).astype(np.uint8)
    w2 = rng.integers(0, 256, size=(E, H, I // 2), dtype=np.uint8)
    s2 = (127 - 7 + rng.integers(0, 3, size=(E, H, I // 32))).astype(np.uint8)
    offsets = np.zeros(E + 1, dtype=np.uint32)
    offsets[1:] = np.cumsum(rows)
    expert_ids = np.arange(E, dtype=np.int32)
    usage = np.array([rows.max(), E], dtype=np.uint32)

    device = Accelerator()
    device_ref = DeviceRef(device.label, device.id)
    cpu_ref = DeviceRef.CPU()
    e8, u8 = DType.float8_e8m0fnu, DType.uint8
    input_types = [
        TensorType(DType.float8_e4m3fn, (total, H), device=device_ref),
        TensorType(e8, (total, H // 32), device=device_ref),
        TensorType(u8, (E, N1, H // 2), device=device_ref),
        TensorType(e8, (E, N1, H // 32), device=device_ref),
        TensorType(u8, (E, H, I // 2), device=device_ref),
        TensorType(e8, (E, H, I // 32), device=device_ref),
        TensorType(DType.uint32, (E + 1,), device=device_ref),
        TensorType(DType.int32, (E,), device=device_ref),
        TensorType(DType.uint32, (2,), device=cpu_ref),
    ]
    with Graph("mega_ffn_amd_a8w4_ab", input_types=input_types) as graph:
        (x_t, sx_t, w1_t, s1_t, w2_t, s2_t, off_t, ids_t, usage_t) = (
            v.tensor for v in graph.inputs
        )
        gate_up = grouped_dynamic_block_scaled_matmul_amd(
            x_t, w1_t, sx_t, s1_t, off_t, ids_t, usage_t
        )
        q, q_scales = swiglu_quantize_dynamic_block_scaled(
            gate_up,
            SITU,
            tensor_sf=1.0,
            sf_vector_size=32,
            scales_type=e8,
            out_type=DType.float8_e4m3fn,
        )
        y_ref = grouped_dynamic_block_scaled_matmul_amd(
            q, w2_t, q_scales, s2_t, off_t, ids_t, usage_t
        )
        words = amd_mega_ffn_workspace_words(E, BAND, total, grids=2)
        if regime == "trap":
            # decode regions of one slot each, which the MegaFFN refuses
            words -= 2 * (max(E + BAND // 16, 1024) - 1) * 32
        workspace = ops.buffer_create(
            BufferType(DType.uint32, shape=[words], device=device_ref),
            init_value=0,
        )
        y = grouped_ffn_block_scaled_amd(
            x_t,
            preshuffle_block_scaled_b(w1_t),
            sx_t,
            s1_t,
            preshuffle_block_scaled_b(w2_t),
            s2_t,
            off_t,
            ids_t,
            usage_t,
            workspace,
            ops.constant(total, dtype=DType.uint32, device=cpu_ref),
            scales_max_padded_m=0,
            decode_grid_m_cap=0,
            decode_grid_m_rows=0,
            mega_band_m=TOP_K if regime == "prefill" else BAND,
            mega_max_rows=BAND,
            activation=SITU,
        )
        # bf16 -> f32 is exact, so equal f32 means equal bf16 bits
        graph.output(y_ref.cast(DType.float32), y.cast(DType.float32))

    compiled = InferenceSession(devices=[device]).load(graph)
    cpu = Accelerator.cpu()

    def dev(a: np.ndarray, dtype: DType | None = None) -> Buffer:
        b = Buffer.from_numpy(a)
        if dtype is not None:
            b = b.view(dtype)
        return b.to(device)

    args = [
        dev(fp8, DType.float8_e4m3fn),
        dev(sx, e8),
        dev(w1),
        dev(s1, e8),
        dev(w2),
        dev(s2, e8),
        dev(offsets),
        dev(expert_ids),
        Buffer.from_numpy(usage),
    ]
    if regime == "trap":
        with pytest.raises(Exception, match="MegaFFN workspace holds"):
            compiled.execute(*args)
        return
    for execution in range(3):
        y_ref_buf, y_buf = compiled.execute(*args)
        y_ref_np = np.from_dlpack(y_ref_buf.to(cpu))
        y_np = np.from_dlpack(y_buf.to(cpu))
        assert np.count_nonzero(y_ref_np) > y_ref_np.size // 2
        bad = int(np.count_nonzero(y_ref_np != y_np))
        assert bad == 0, (
            f"execution {execution}: {bad} of {y_np.size} bf16 outputs"
            f" differ from the graph ops (regime={regime}, tokens={tokens})"
        )


@pytest.mark.skipif(
    accelerator_api() != "hip", reason="the AMD MoE FFN is gfx950-only"
)
def test_mega_ffn_amd_a8w4_workspace_survives_varying_rows() -> None:
    """One persistent workspace through steps of varying size, as tensor
    parallelism serves them: the routed rows cross 16384, past which the
    prefill region needs more than its minimum 1024 slots. Each output
    matches the graph ops bit for bit (the regions used to move there, and
    the next prefill waited on counters forever)."""
    rng = np.random.default_rng(7)
    w1 = rng.integers(0, 256, size=(E, N1, H // 2), dtype=np.uint8)
    s1 = (127 - 7 + rng.integers(0, 3, size=(E, N1, H // 32))).astype(np.uint8)
    w2 = rng.integers(0, 256, size=(E, H, I // 2), dtype=np.uint8)
    s2 = (127 - 7 + rng.integers(0, 3, size=(E, H, I // 32))).astype(np.uint8)
    capacity = 1100 * TOP_K

    device = Accelerator()
    device_ref = DeviceRef(device.label, device.id)
    cpu_ref = DeviceRef.CPU()
    e8, u8 = DType.float8_e8m0fnu, DType.uint8
    input_types = [
        TensorType(DType.float8_e4m3fn, ("rows", H), device=device_ref),
        TensorType(e8, ("rows", H // 32), device=device_ref),
        TensorType(u8, (E, N1, H // 2), device=device_ref),
        TensorType(e8, (E, N1, H // 32), device=device_ref),
        TensorType(u8, (E, H, I // 2), device=device_ref),
        TensorType(e8, (E, H, I // 32), device=device_ref),
        TensorType(DType.uint32, (E + 1,), device=device_ref),
        TensorType(DType.int32, (E,), device=device_ref),
        TensorType(DType.uint32, (2,), device=cpu_ref),
        TensorType(DType.uint32, (), device=cpu_ref),
    ]
    with Graph("mega_ffn_amd_a8w4_rows", input_types=input_types) as graph:
        (x_t, sx_t, w1_t, s1_t, w2_t, s2_t, off_t, ids_t, usage_t, m_t) = (
            v.tensor for v in graph.inputs
        )
        gate_up = grouped_dynamic_block_scaled_matmul_amd(
            x_t, w1_t, sx_t, s1_t, off_t, ids_t, usage_t
        )
        q, q_scales = swiglu_quantize_dynamic_block_scaled(
            gate_up,
            SITU,
            tensor_sf=1.0,
            sf_vector_size=32,
            scales_type=e8,
            out_type=DType.float8_e4m3fn,
        )
        y_ref = grouped_dynamic_block_scaled_matmul_amd(
            q, w2_t, q_scales, s2_t, off_t, ids_t, usage_t
        )
        workspace = ops.buffer_create(
            BufferType(
                DType.uint32,
                shape=[
                    amd_mega_ffn_workspace_words(E, BAND, capacity, grids=2)
                ],
                device=device_ref,
            ),
            init_value=0,
        )
        y = grouped_ffn_block_scaled_amd(
            x_t,
            preshuffle_block_scaled_b(w1_t),
            sx_t,
            s1_t,
            preshuffle_block_scaled_b(w2_t),
            s2_t,
            off_t,
            ids_t,
            usage_t,
            workspace,
            m_t,
            scales_max_padded_m=0,
            decode_grid_m_cap=0,
            decode_grid_m_rows=0,
            mega_band_m=BAND,
            mega_max_rows=BAND,
            activation=SITU,
        )
        graph.output(y_ref.cast(DType.float32), y.cast(DType.float32))

    compiled = InferenceSession(devices=[device]).load(graph)
    cpu = Accelerator.cpu()

    def dev(a: np.ndarray, dtype: DType | None = None) -> Buffer:
        b = Buffer.from_numpy(a)
        if dtype is not None:
            b = b.view(dtype)
        return b.to(device)

    weights = [dev(w1), dev(s1, e8), dev(w2), dev(s2, e8)]
    for step, tokens in enumerate((1000, 1100, 9, 1000)):
        rows = _rows(rng, tokens)
        total = int(rows.sum())
        fp8 = rng.integers(0, 256, size=(total, H), dtype=np.uint8)
        fp8[(fp8 & 0x7F) == 0x7F] ^= 1
        sx = (127 - 9 + rng.integers(0, 3, size=(total, H // 32))).astype(
            np.uint8
        )
        offsets = np.zeros(E + 1, dtype=np.uint32)
        offsets[1:] = np.cumsum(rows)
        y_ref_buf, y_buf = compiled.execute(
            dev(fp8, DType.float8_e4m3fn),
            dev(sx, e8),
            *weights,
            dev(offsets),
            dev(np.arange(E, dtype=np.int32)),
            Buffer.from_numpy(np.array([rows.max(), E], dtype=np.uint32)),
            Buffer.from_numpy(np.array(total, dtype=np.uint32)),
        )
        y_ref_np = np.from_dlpack(y_ref_buf.to(cpu))
        y_np = np.from_dlpack(y_buf.to(cpu))
        bad = int(np.count_nonzero(y_ref_np != y_np))
        assert bad == 0, (
            f"step {step} ({tokens} tokens, {total} rows): {bad} of"
            f" {y_np.size} bf16 outputs differ from the graph ops"
        )


def _preb(w: np.ndarray) -> np.ndarray:
    """``[E, N, K_BYTES]`` in preb's B layout: per 16 rows and 64 K bytes,
    ``(klane, nlane, 16)``."""
    e, n, kb = w.shape
    return (
        w.reshape(e, n // 16, 16, kb // 64, 4, 16)
        .transpose(0, 1, 3, 4, 2, 5)
        .reshape(e, n, kb)
    )


@pytest.mark.parametrize(
    "axis, shards, on_gpu", [(0, 3, False), (1, 3, False), (0, 1, True)]
)
def test_stack_a8w4_mega_ffn_b(axis: int, shards: int, on_gpu: bool) -> None:
    """Each shard's stack is its experts' projections, split along ``axis``
    and concatenated, in preb's layout: from the host straight into host
    stacks, and through pinned staging into a GPU's."""
    rng = np.random.default_rng(axis)
    experts, projections = 5, 2 if axis == 0 else 1
    rows, k = (96, 384) if axis == 0 else (64, 576)
    srcs = [
        rng.integers(0, 256, (rows, k), dtype=np.uint8)
        for _ in range(experts * projections)
    ]
    cpu_ref = DeviceRef.CPU()
    session_devices: list[CPU | Accelerator] = [CPU()]
    if on_gpu:
        gpu = Accelerator()
        session_devices.append(gpu)
        devices = [DeviceRef(gpu.label, gpu.id)]
    else:
        devices = [cpu_ref] * shards
    input_types = [TensorType(DType.uint8, (rows, k), device=cpu_ref)] * len(
        srcs
    )
    with Graph("stack_a8w4_b", input_types=input_types) as graph:
        graph.output(
            *stack_a8w4_mega_ffn_b(
                [v.tensor for v in graph.inputs], experts, devices, axis
            )
        )
    outs = (
        InferenceSession(devices=session_devices)
        .load(graph)
        .execute(*(Buffer.from_numpy(a) for a in srcs))
    )
    for d, out in enumerate(outs):
        shard = [np.split(a, len(devices), axis=axis)[d] for a in srcs]
        want = _preb(
            np.stack(
                [
                    np.concatenate(
                        shard[e * projections : (e + 1) * projections]
                    )
                    for e in range(experts)
                ]
            )
        )
        got = np.from_dlpack(out.to(CPU()))
        assert np.array_equal(got, want), f"shard {d} differs"
