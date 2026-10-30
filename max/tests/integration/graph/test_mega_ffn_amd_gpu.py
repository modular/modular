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

"""Graph-level A/B of the AMD MXFP8 MoE FFN op (`mo.grouped.ffn.block.scaled.amd`)
against the production two-op chain it replaces (gate/up with the fused
SwiGLU + MXFP8 requantize, then down), byte for byte, and of the
blockwise-FP8 op (`mo.grouped.ffn.blockwise.fp8.amd`) against the GLM-5.3
chain (grouped FP8 gate/up into bf16, the SwiGLU fused into the row-bounded
quantize, grouped FP8 down).

In the decode band the op runs the one-launch MegaFFN, whose counters live in
a persistent graph buffer zeroed once at load; the graph executes several
times so a stale count across executions would show. Above its prefill band
it runs the prefill MegaFFN (or the two prefill GEMMs), here on partial
128-row tiles and an empty expert.
Elsewhere the op runs the chain itself.

NOTE: AMD gfx950 only.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from max import functional as F
from max.driver import Accelerator, Buffer, accelerator_api
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    Graph,
    TensorType,
    TensorValue,
    ops,
)
from max.nn import Module
from max.nn.common_layers.mesh_axis import TP
from max.nn.common_layers.quant_kernels import preshuffle_blockwise_fp8_b
from max.nn.common_layers.quant_moe import ExpertParallelMoE
from max.nn.common_layers.quant_ops import (
    EPDispatchPayload,
    amd_blockwise_mega_band_m,
    amd_blockwise_mega_max_rows,
    grouped_ffn_blockwise_fp8_amd_ep,
)
from max.nn.common_layers.quant_tensor import (
    FP8BlockActivation,
    FP8BlockTensor,
)
from max.nn.legacy.comm.ep import EPBatchManager, EPConfig
from max.nn.legacy.kernels import (
    BLOCKWISE_FP8_LANE_BYTES,
    amd_mega_ffn_workspace_words,
    block_scaled_preshuffle_grouped_scale_4d,
    grouped_dynamic_block_scaled_matmul_amd,
    grouped_dynamic_block_scaled_matmul_amd_swiglu_quant,
    grouped_dynamic_scaled_fp8_matmul,
    grouped_ffn_block_scaled_amd,
    grouped_ffn_blockwise_fp8_amd,
    preshuffle_block_scaled_b,
    preshuffle_block_scaled_b_scales,
    quantize_dynamic_scaled_float8,
)
from max.nn.legacy.moe.moe_fp8 import MoEQuantized
from max.nn.legacy.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)
from max.pipelines.architectures.glm5_next_modulev3.layers.mlp import (
    Glm5NextExpertParallelMoE,
)
from max.tensor import Tensor, default_device, default_dtype
from max.tensor.compilation import CompiledCallable
from max.tensor.realization_context import ensure_context
from max.tensor.sharding import DeviceMesh

H = 6144
I = 3072
SLOT_ROWS = 32


def _fp8_bytes(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    """E4M3 bytes of magnitude ~0.5..1.9 with random sign (no NaN codes)."""
    v = rng.integers(0, 16, size=shape, dtype=np.uint8)
    return (0x30 + (v & 7)) | ((v & 8) << 4)


def _scale_bytes(
    rng: np.random.Generator, shape: tuple[int, ...], lo: int
) -> np.ndarray:
    return (127 + lo + rng.integers(0, 3, size=shape)).astype(np.uint8)


@pytest.mark.parametrize("n_nodes, fused", [(1, True), (2, False)])
def test_mega_ffn_amd_single_node_only(n_nodes: int, fused: bool) -> None:
    """Multi-node EP keeps the chain: remote nodes' rows are unbounded."""
    moe = SimpleNamespace(
        hidden_dim=H,
        moe_dim=I,
        _ep_batch_manager=SimpleNamespace(
            config=SimpleNamespace(n_nodes=n_nodes)
        ),
        _can_fuse_swiglu_mxfp8=lambda: True,
    )
    assert MoEQuantized._can_fuse_mega_ffn_amd(cast(MoEQuantized, moe)) is fused


@pytest.mark.skipif(
    accelerator_api() != "hip", reason="the AMD MoE FFN is gfx950-only"
)
@pytest.mark.parametrize(
    "regime, i",
    [
        ("mega", I),
        ("prefill_mega", I),
        ("prefill_gemm", I),
        ("chain", I),
        # below the prefill kernels' K floor (512): the op takes the chain
        ("prefill_mega", 256),
    ],
)
def test_mega_ffn_amd_matches_chain(regime: str, i: int) -> None:
    n1 = 2 * i
    tokens_per_expert = [16, 0, 21, 9]
    n_experts = len(tokens_per_expert)
    total = sum(tokens_per_expert)
    max_tok = max(tokens_per_expert)
    assert max_tok <= SLOT_ROWS
    rng = np.random.default_rng(11)
    x = _fp8_bytes(rng, (total, H))
    sx = _scale_bytes(rng, (total, H // 32), -4)
    w1 = _fp8_bytes(rng, (n_experts, n1, H))
    s1 = _scale_bytes(rng, (n_experts, n1, H // 32), -4)
    w2 = _fp8_bytes(rng, (n_experts, H, i))
    s2 = _scale_bytes(rng, (n_experts, H, i // 32), -2)
    offsets = np.zeros(n_experts + 1, dtype=np.uint32)
    offsets[1:] = np.cumsum(tokens_per_expert)
    expert_ids = np.arange(n_experts, dtype=np.int32)
    usage = np.array([max_tok, n_experts], dtype=np.uint32)

    device = Accelerator()
    device_ref = DeviceRef(device.label, device.id)
    cpu_ref = DeviceRef.CPU()
    session = InferenceSession(devices=[device])
    e8 = DType.float8_e8m0fnu
    f8 = DType.float8_e4m3fn
    input_types = [
        TensorType(f8, (total, H), device=device_ref),
        TensorType(e8, (total, H // 32), device=device_ref),
        TensorType(f8, (n_experts, n1, H), device=device_ref),
        TensorType(e8, (n_experts, n1, H // 32), device=device_ref),
        TensorType(f8, (n_experts, H, i), device=device_ref),
        TensorType(e8, (n_experts, H, i // 32), device=device_ref),
        TensorType(DType.uint32, (n_experts + 1,), device=device_ref),
        TensorType(DType.int32, (n_experts,), device=device_ref),
        TensorType(DType.uint32, (2,), device=cpu_ref),
    ]
    decode_band = regime == "mega"
    # decode band: estimated rows under the cap; otherwise the gate is off
    cap = 128 if decode_band else 0
    with Graph("mega_ffn_amd_ab", input_types=input_types) as graph:
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

        q, q_scales = grouped_dynamic_block_scaled_matmul_amd_swiglu_quant(
            x_t,
            w1_pre,
            sx_slot,
            s1_pre,
            off_t,
            ids_t,
            usage_t,
            estimated_total_m=est_m,
            a_scales_max_padded_m=SLOT_ROWS,
            decode_grid_m_cap=cap,
            decode_grid_m_rows=max_tok,
            out_scales_max_padded_m=SLOT_ROWS,
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
            decode_grid_m_cap=cap,
            decode_grid_m_rows=max_tok,
        )
        workspace = ops.buffer_create(
            BufferType(
                DType.uint32,
                shape=[amd_mega_ffn_workspace_words(n_experts, total, total)],
                device=device_ref,
            ),
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
            decode_grid_m_cap=cap,
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
        dev(x, f8),
        dev(sx, e8),
        dev(w1, f8),
        dev(s1, e8),
        dev(w2, f8),
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


BW_H = 6144
BW_I = 2048
BW_ROWS = 64
"""Rows of the EP receive buffer: its tail past the experts' rows is
garbage neither path may read."""
BW_ROWS_REUSE = 640
"""A receive buffer large enough that the decode MegaFFN reuses the op's act
output as its scratch, as at production shapes."""


@pytest.mark.parametrize("n_gpus", [2, 4, 8])
@pytest.mark.parametrize("n_experts", [33, 145])
def test_blockwise_mega_max_rows_bounds_the_band(
    n_gpus: int, n_experts: int
) -> None:
    """The largest step inside the decode band, every routed row on one
    GPU, stays within the decode MegaFFN's row bound."""
    top_k = 8
    band = amd_blockwise_mega_band_m(n_experts)
    tokens = band * n_gpus // top_k
    assert tokens * top_k // n_gpus <= band
    assert (
        amd_blockwise_mega_max_rows(band, n_gpus, top_k)
        >= tokens * top_k + tokens
    )


@pytest.mark.skipif(
    accelerator_api() != "hip", reason="the AMD MoE FFN is gfx950-only"
)
@pytest.mark.parametrize(
    "n_nodes, accum, limit, hidden, fmt, block, fused",
    [
        (1, None, 0.0, BW_H, QuantFormat.BLOCKSCALED_FP8, (128, 128), True),
        (
            1,
            DType.float32,
            10.0,
            4096,
            QuantFormat.BLOCKSCALED_FP8,
            (128, 128),
            True,
        ),
        (
            1,
            DType.float32,
            0.0,
            BW_H,
            QuantFormat.BLOCKSCALED_FP8,
            (128, 128),
            True,
        ),
        (2, None, 0.0, BW_H, QuantFormat.BLOCKSCALED_FP8, (128, 128), False),
        (1, None, 10.0, 4096, QuantFormat.BLOCKSCALED_FP8, (128, 128), True),
        (
            1,
            DType.bfloat16,
            0.0,
            BW_H,
            QuantFormat.BLOCKSCALED_FP8,
            (128, 128),
            False,
        ),
        (1, None, 0.0, 7168, QuantFormat.BLOCKSCALED_FP8, (128, 128), False),
        (1, None, 0.0, BW_H, QuantFormat.MXFP8, (128, 128), False),
        (1, None, 0.0, BW_H, QuantFormat.BLOCKSCALED_FP8, (1, 128), False),
    ],
)
def test_blockwise_mega_ffn_amd_gate(
    n_nodes: int,
    accum: DType | None,
    limit: float,
    hidden: int,
    fmt: QuantFormat,
    block: tuple[int, int],
    fused: bool,
) -> None:
    """Single node, a validated shape and blockwise scales, and a bf16 or
    fp32 SwiGLU, clamped or not."""
    moe = SimpleNamespace(
        hidden_dim=hidden,
        moe_dim=BW_I,
        ep_batch_manager=SimpleNamespace(
            config=SimpleNamespace(n_nodes=n_nodes)
        ),
        quant_config=SimpleNamespace(
            format=fmt,
            weight_scale=SimpleNamespace(block_size=block),
        ),
        gate_up_accum_dtype=accum,
        swiglu_limit=limit,
    )
    # The op's registration is read off the current graph's kernel library.
    with Graph("blockwise_mega_ffn_amd_gate", input_types=[]):
        eligible = ExpertParallelMoE._blockwise_mega_ffn_amd_eligible(
            cast(ExpertParallelMoE, moe)
        )
    assert eligible is fused


BW_ROUTING = (16, 0, 21, 9)
BW_ROUTING_T1 = (1, 0, 1, 1)
"""A one-token step: the shared expert's row and one row per routed expert."""
BW_BAND = 256
"""The decode band every A/B graph is built with; the regime is an input."""


class _BlockwiseAB:
    """One compiled A/B graph and the inputs it runs on.

    The chain, the op and, at ``BW_ROWS``, the ModuleV3 EP binding on the
    experts permuted the way the layer stacks them, all at one structure. The
    regime is the ``estimated_total_m`` input.
    """

    def __init__(
        self,
        session: InferenceSession,
        act_fp32: bool,
        swiglu_limit: float,
        rows: int,
        routing: tuple[int, ...],
    ) -> None:
        tokens_per_expert = list(routing)
        n_experts = len(tokens_per_expert)
        self.total = total = sum(tokens_per_expert)
        assert total <= rows
        h, i, n1 = BW_H, BW_I, 2 * BW_I
        rng = np.random.default_rng(13)
        self.x = _fp8_bytes(rng, (rows, h))
        self.sx = rng.uniform(0.1, 0.3, (h // 128, rows)).astype(np.float32)
        self.w1 = _fp8_bytes(rng, (n_experts, n1, h))
        # gate/up of magnitude up to ~20, so the limit of 10 clamps both
        self.s1 = rng.uniform(
            0.3, 0.9, (n_experts, n1 // 128, h // 128)
        ).astype(np.float32)
        self.w2 = _fp8_bytes(rng, (n_experts, h, i))
        self.s2 = rng.uniform(
            0.005, 0.02, (n_experts, h // 128, i // 128)
        ).astype(np.float32)
        self.offsets = np.zeros(n_experts + 1, dtype=np.uint32)
        self.offsets[1:] = np.cumsum(tokens_per_expert)
        self.expert_ids = np.arange(n_experts, dtype=np.int32)
        self.usage = np.array(
            [max(tokens_per_expert), n_experts], dtype=np.uint32
        )
        self.with_ep = rows == BW_ROWS

        self.device = device = session.devices[0]
        device_ref = DeviceRef(device.label, device.id)
        cpu_ref = DeviceRef.CPU()
        f8 = DType.float8_e4m3fn
        f32 = DType.float32
        input_types = [
            TensorType(f8, (rows, h), device=device_ref),
            TensorType(f32, (h // 128, rows), device=device_ref),
            TensorType(f8, (n_experts, n1, h), device=device_ref),
            TensorType(
                f32, (n_experts, n1 // 128, h // 128), device=device_ref
            ),
            TensorType(f8, (n_experts, h, i), device=device_ref),
            TensorType(f32, (n_experts, h // 128, i // 128), device=device_ref),
            TensorType(DType.uint32, (n_experts + 1,), device=device_ref),
            TensorType(DType.int32, (n_experts,), device=device_ref),
            TensorType(DType.uint32, (2,), device=cpu_ref),
            TensorType(DType.uint32, (), device=cpu_ref),
        ]
        input_spec = InputScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            origin=ScaleOrigin.DYNAMIC,
            dtype=f32,
            block_size=(1, 128),
        )
        weight_spec = WeightScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            dtype=f32,
            block_size=(128, 128),
        )
        with Graph(
            "mega_ffn_amd_blockwise_ab", input_types=input_types
        ) as graph:
            (
                x_t,
                sx_t,
                w1_t,
                s1_t,
                w2_t,
                s2_t,
                off_t,
                ids_t,
                usage_t,
                est_t,
            ) = (v.tensor for v in graph.inputs)
            gate_up = grouped_dynamic_scaled_fp8_matmul(
                x_t,
                w1_t,
                sx_t,
                s1_t,
                off_t,
                ids_t,
                usage_t,
                input_spec,
                weight_spec,
                static_grid_z=True,
            )
            if act_fp32:
                gate_up = gate_up.cast(f32)
            gate, up = gate_up[:, :i], gate_up[:, i:]
            if swiglu_limit:
                gate = ops.min(gate, swiglu_limit)
                up = ops.min(ops.max(up, -swiglu_limit), swiglu_limit)
            q, q_scales = quantize_dynamic_scaled_float8(
                ops.silu(gate) * up,
                input_spec,
                weight_spec,
                group_size_or_per_token=128,
                scales_type=f32,
                out_type=f8,
                row_offsets=off_t,
            )
            y_ref = grouped_dynamic_scaled_fp8_matmul(
                q,
                w2_t,
                q_scales,
                s2_t,
                off_t,
                ids_t,
                usage_t,
                input_spec,
                weight_spec,
                static_grid_z=True,
            )
            workspace = ops.buffer_create(
                BufferType(
                    DType.uint32,
                    shape=[
                        amd_mega_ffn_workspace_words(
                            n_experts, total, rows, grids=2
                        )
                    ],
                    device=device_ref,
                ),
                init_value=0,
            )
            y = grouped_ffn_blockwise_fp8_amd(
                x_t,
                preshuffle_block_scaled_b(w1_t, BLOCKWISE_FP8_LANE_BYTES),
                sx_t,
                s1_t,
                preshuffle_block_scaled_b(w2_t, BLOCKWISE_FP8_LANE_BYTES),
                s2_t,
                off_t,
                ids_t,
                usage_t,
                workspace,
                est_t,
                mega_band_m=BW_BAND,
                mega_max_rows=total,
                activation_fp32=act_fp32,
                swiglu_limit=swiglu_limit,
            )
            outputs = [y_ref[:total].cast(f32), y[:total].cast(f32)]
            if self.with_ep:
                with ensure_context():
                    t = Tensor.from_graph_value
                    y_ep = grouped_ffn_blockwise_fp8_amd_ep(
                        FP8BlockActivation(data=t(x_t), scales=t(sx_t)),
                        FP8BlockTensor(
                            weight=preshuffle_blockwise_fp8_b(t(w1_t)),
                            weight_scale_inv=t(s1_t),
                        ),
                        FP8BlockTensor(
                            weight=preshuffle_blockwise_fp8_b(t(w2_t)),
                            weight_scale_inv=t(s2_t),
                        ),
                        t(off_t),
                        t(ids_t),
                        t(usage_t),
                        t(est_t),
                        BW_BAND,
                        total,
                        activation_fp32=act_fp32,
                        swiglu_limit=swiglu_limit,
                    )
                    outputs.append(TensorValue(y_ep)[:total].cast(f32))
            graph.output(*outputs)
        self.model = session.load(graph)


BW_VARIANTS = {
    "bf16": (False, 0.0, BW_ROWS, BW_ROUTING),
    "bf16_limit": (False, 10.0, BW_ROWS, BW_ROUTING),
    "fp32_limit": (True, 10.0, BW_ROWS, BW_ROUTING),
    "fp32": (True, 0.0, BW_ROWS, BW_ROUTING),
    "bf16_reuse": (False, 0.0, BW_ROWS_REUSE, BW_ROUTING),
    "bf16_limit_reuse": (False, 10.0, BW_ROWS_REUSE, BW_ROUTING),
    "bf16_t1": (False, 0.0, BW_ROWS, BW_ROUTING_T1),
    "bf16_limit_t1": (False, 10.0, BW_ROWS, BW_ROUTING_T1),
}
"""Each A/B graph's SwiGLU precision, clamp, receive rows and routing."""


@pytest.fixture(scope="module")
def blockwise_ab() -> dict[str, _BlockwiseAB]:
    """Compiles every A/B graph before a test puts data on the device."""
    if accelerator_api() != "hip":
        pytest.skip("the AMD MoE FFN is gfx950-only")
    session = InferenceSession(devices=[Accelerator()])
    return {
        name: _BlockwiseAB(session, *variant)
        for name, variant in BW_VARIANTS.items()
    }


@pytest.mark.skipif(
    accelerator_api() != "hip", reason="the AMD MoE FFN is gfx950-only"
)
@pytest.mark.parametrize(
    "regime, variant",
    [
        ("mega_narrow", "bf16"),
        ("mega_wide", "bf16"),
        ("prefill_mega", "bf16"),
        ("mega_narrow", "bf16_limit"),
        ("mega_wide", "bf16_limit"),
        ("prefill_mega", "bf16_limit"),
        ("mega_narrow", "fp32_limit"),
        ("mega_wide", "fp32_limit"),
        ("prefill_mega", "fp32_limit"),
        ("mega_wide", "fp32"),
        ("prefill_mega", "fp32"),
        ("mega_narrow", "bf16_reuse"),
        ("mega_wide", "bf16_limit_reuse"),
        ("mega_narrow", "bf16_t1"),
        ("mega_narrow", "bf16_limit_t1"),
    ],
)
def test_mega_ffn_amd_blockwise_matches_chain(
    blockwise_ab: dict[str, _BlockwiseAB], regime: str, variant: str
) -> None:
    """GLM-5.3's bf16 SwiGLU, GLM-5.3-Flash's bf16 one clamped at 10, and
    fp32 ones clamped and not; a receive buffer large enough for the decode
    MegaFFN to reuse the act output as scratch; a one-token step."""
    ab = blockwise_ab[variant]
    # the regime: estimated rows up to 16 decode on one workgroup a CU,
    # and rows past the band run the prefill MegaFFN
    est = {
        "mega_narrow": 16,
        "mega_wide": ab.total,
        "prefill_mega": BW_BAND + 1,
    }[regime]
    if regime == "mega_wide":
        assert 16 < est <= BW_BAND
    cpu = Accelerator.cpu()
    f8 = DType.float8_e4m3fn

    def dev(a: np.ndarray, dtype: DType | None = None) -> Buffer:
        b = Buffer.from_numpy(a)
        if dtype is not None:
            b = b.view(dtype)
        return b.to(ab.device)

    args = [
        dev(ab.x, f8),
        dev(ab.sx),
        dev(ab.w1, f8),
        dev(ab.s1),
        dev(ab.w2, f8),
        dev(ab.s2),
        dev(ab.offsets),
        dev(ab.expert_ids),
        Buffer.from_numpy(ab.usage),
        Buffer.from_numpy(np.array(est, dtype=np.uint32)),
    ]
    for execution in range(3):
        y_ref_buf, *outs = ab.model.execute(*args)
        assert len(outs) == 1 + ab.with_ep
        y_ref_np = np.from_dlpack(y_ref_buf.to(cpu))
        assert np.count_nonzero(y_ref_np) > y_ref_np.size // 2
        for path, buf in zip(("op", "EP binding"), outs, strict=False):
            y_np = np.from_dlpack(buf.to(cpu))
            bad = int(np.count_nonzero(y_ref_np != y_np))
            assert bad == 0, (
                f"execution {execution}: {bad} of {y_np.size} bf16 outputs"
                f" of the {path} differ from the chain (regime={regime},"
                f" variant={variant})"
            )


def _blockwise_values(
    rows: int, n_experts: int, workspace_words: int
) -> tuple[Graph, list[TensorValue], BufferValue]:
    """A graph with the blockwise op's inputs at the test's GLM-5.3 shape."""
    h, i = BW_H, BW_I
    device_ref = DeviceRef.GPU()
    cpu_ref = DeviceRef.CPU()
    f8, f32 = DType.float8_e4m3fn, DType.float32
    graph = Graph(
        "mega_ffn_amd_blockwise_raises",
        input_types=[
            TensorType(f8, (rows, h), device=device_ref),
            TensorType(f32, (h // 128, rows), device=device_ref),
            TensorType(f8, (n_experts, 2 * i, h), device=device_ref),
            TensorType(
                f32, (n_experts, 2 * i // 128, h // 128), device=device_ref
            ),
            TensorType(f8, (n_experts, h, i), device=device_ref),
            TensorType(f32, (n_experts, h // 128, i // 128), device=device_ref),
            TensorType(DType.uint32, (n_experts + 1,), device=device_ref),
            TensorType(DType.int32, (n_experts,), device=device_ref),
            TensorType(DType.uint32, (2,), device=cpu_ref),
        ],
    )
    with graph:
        values = [v.tensor for v in graph.inputs]
        workspace = ops.buffer_create(
            BufferType(
                DType.uint32, shape=[workspace_words], device=device_ref
            ),
            init_value=0,
        )
    return graph, values, workspace


@pytest.mark.parametrize(
    "operand, shape",
    [
        ("down_weight", (4, BW_H, BW_I // 2)),
        ("down_weight", (4, BW_H // 2, BW_I)),
        ("down_scales", (4, BW_H // 128, BW_I // 256)),
        ("a_scales", (BW_H // 256, BW_ROWS)),
        ("hidden_states", (BW_ROWS, BW_H // 2)),
    ],
)
def test_blockwise_mega_ffn_amd_rejects_mismatched_shapes(
    operand: str, shape: tuple[int, ...]
) -> None:
    """Every operand must match ``gate_up_weight``, which the op sizes from."""
    f8, f32 = DType.float8_e4m3fn, DType.float32
    dev = DeviceRef.CPU()
    types = {
        "hidden_states": TensorType(f8, (BW_ROWS, BW_H), device=dev),
        "gate_up_weight": TensorType(f8, (4, 2 * BW_I, BW_H), device=dev),
        "a_scales": TensorType(f32, (BW_H // 128, BW_ROWS), device=dev),
        "gate_up_scales": TensorType(
            f32, (4, 2 * BW_I // 128, BW_H // 128), device=dev
        ),
        "down_weight": TensorType(f8, (4, BW_H, BW_I), device=dev),
        "down_scales": TensorType(
            f32, (4, BW_H // 128, BW_I // 128), device=dev
        ),
    }
    types[operand] = TensorType(types[operand].dtype, shape, device=dev)
    names = list(types)
    with Graph("bw_shapes", input_types=list(types.values())) as graph:
        v = dict(zip(names, (x.tensor for x in graph.inputs), strict=True))
        with pytest.raises(ValueError, match=operand):
            grouped_ffn_blockwise_fp8_amd(
                v["hidden_states"],
                v["gate_up_weight"],
                v["a_scales"],
                v["gate_up_scales"],
                v["down_weight"],
                v["down_scales"],
                ops.constant(np.zeros(5, np.uint32), DType.uint32, dev),
                ops.constant(np.zeros(4, np.int32), DType.int32, dev),
                ops.constant(np.zeros(2, np.uint32), DType.uint32, dev),
                ops.buffer_create(
                    BufferType(DType.uint32, shape=[8], device=dev)
                ),
                ops.constant(0, DType.uint32, dev),
                mega_band_m=256,
                mega_max_rows=BW_ROWS,
            )


@pytest.mark.skipif(
    accelerator_api() != "hip", reason="the AMD MoE FFN is gfx950-only"
)
@pytest.mark.parametrize("fault", ["workspace", "act_scales"])
def test_mega_ffn_amd_blockwise_rejects_small_buffers(fault: str) -> None:
    """The op raises on a workspace or prefill act-scale buffer too small for
    the launch rather than writing past it."""
    tokens_per_expert = [16, 0, 21, 9]
    n_experts = len(tokens_per_expert)
    rows = BW_ROWS
    words = amd_mega_ffn_workspace_words(n_experts, rows, rows, grids=2)
    graph, values, workspace = _blockwise_values(
        rows, n_experts, 8 if fault == "workspace" else words
    )
    x_t, sx_t, w1_t, s1_t, w2_t, s2_t, off_t, ids_t, usage_t = values
    with graph:
        cpu_ref = DeviceRef.CPU()
        # the prefill regime, so both buffers are in play
        args: list[TensorValue | BufferValue] = [
            x_t,
            w1_t,
            sx_t,
            s1_t,
            w2_t,
            s2_t,
            off_t,
            ids_t,
            workspace,
            usage_t[1],
            ops.constant(rows, DType.uint32, cpu_ref),
            ops.constant(0, DType.uint32, cpu_ref),
            ops.constant(rows, DType.uint32, cpu_ref),
        ]
        scale_rows = (
            1 if fault == "act_scales" else rows + 128 * (n_experts + 1)
        )
        outputs = ops.inplace_custom(
            "mo.grouped.ffn.blockwise.fp8.amd",
            device=x_t.device,
            values=args,
            out_types=[
                TensorType(DType.bfloat16, [rows, BW_H], device=x_t.device),
                TensorType(
                    DType.float8_e4m3fn, [rows, BW_I], device=x_t.device
                ),
                TensorType(
                    DType.float32, [scale_rows, BW_I // 128], device=x_t.device
                ),
            ],
            parameters={
                "scales_max_padded_m": rows,
                "act_fp32": False,
                "swiglu_limit": 0.0,
            },
        )
        graph.output(outputs[0].tensor)
    device = Accelerator()
    compiled = InferenceSession(devices=[device]).load(graph)
    rng = np.random.default_rng(5)
    offsets = np.zeros(n_experts + 1, dtype=np.uint32)
    offsets[1:] = np.cumsum(tokens_per_expert)

    def dev(a: np.ndarray, dtype: DType | None = None) -> Buffer:
        b = Buffer.from_numpy(a)
        return (b.view(dtype) if dtype is not None else b).to(device)

    f8 = DType.float8_e4m3fn
    match = "workspace holds" if fault == "workspace" else "act scales hold"
    with pytest.raises(Exception, match=match):
        compiled.execute(
            dev(_fp8_bytes(rng, (rows, BW_H)), f8),
            dev(np.ones((BW_H // 128, rows), np.float32)),
            dev(_fp8_bytes(rng, (n_experts, 2 * BW_I, BW_H)), f8),
            dev(np.ones((n_experts, 2 * BW_I // 128, BW_H // 128), np.float32)),
            dev(_fp8_bytes(rng, (n_experts, BW_H, BW_I)), f8),
            dev(np.ones((n_experts, BW_H // 128, BW_I // 128), np.float32)),
            dev(offsets),
            dev(np.arange(n_experts, dtype=np.int32)),
            Buffer.from_numpy(
                np.array([max(tokens_per_expert), n_experts], dtype=np.uint32)
            ),
        )


class _LocalExperts(Module[..., Tensor]):
    """An EP layer's local expert compute on already-dispatched rows."""

    def __init__(self, moe: ExpertParallelMoE) -> None:
        self.moe = moe

    def forward(self, *args: Tensor) -> Tensor:
        x, sx, offsets, expert_ids, usage, estimated_total_m = args
        payload = EPDispatchPayload(
            tokens=[x],
            expert_start=[offsets],
            expert_ids=[expert_ids],
            usage_stats=[usage],
            scales=[sx],
        )
        (y,) = self.moe._local_compute(payload, None, estimated_total_m)
        return y.cast(DType.float32)


@pytest.mark.skipif(
    accelerator_api() != "hip", reason="the AMD MoE FFN is gfx950-only"
)
@pytest.mark.parametrize("h, swiglu_limit", [(BW_H, 0.0), (4096, 10.0)])
def test_blockwise_mega_ffn_layer_reads_checkpoint_experts(
    monkeypatch: pytest.MonkeyPatch, h: int, swiglu_limit: float
) -> None:
    """A MegaFFN layer compiled the ModuleV3 way, its weights the checkpoint's
    row-major experts as an engine hands them over, matches the chain the
    same layer compiles to without the MegaFFN, bit for bit, in its decode
    and prefill bands. GLM-5.3 and GLM-5.3-Flash, through the EP MoE class
    their models build, so a change there that drops the MegaFFN fails."""
    i, top_k, rows = BW_I, 8, BW_ROWS
    tokens_per_expert = BW_ROUTING
    n_experts, total = len(tokens_per_expert), sum(tokens_per_expert)
    device = Accelerator()
    mesh = DeviceMesh((device,), (1,), (TP,))
    f8, f32 = DType.float8_e4m3fn, DType.float32
    quant_config = QuantConfig(
        input_scale=InputScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            origin=ScaleOrigin.DYNAMIC,
            dtype=f32,
            block_size=(1, 128),
        ),
        weight_scale=WeightScaleSpec(
            granularity=ScaleGranularity.BLOCK, dtype=f32, block_size=(128, 128)
        ),
        mlp_quantized_layers={0},
        attn_quantized_layers={0},
        format=QuantFormat.BLOCKSCALED_FP8,
    )
    ep_config = EPConfig(
        dispatch_dtype=f8,
        combine_dtype=DType.bfloat16,
        hidden_size=h,
        top_k=top_k,
        n_experts=n_experts,
        max_tokens_per_rank=rows,
        n_gpus_per_node=1,
        n_nodes=1,
        dispatch_quant_config=quant_config,
    )
    with F.lazy(), default_device(mesh), default_dtype(DType.bfloat16):
        moe = Glm5NextExpertParallelMoE(
            hidden_dim=h,
            num_experts=n_experts,
            num_experts_per_token=top_k,
            moe_dim=i,
            quant_config=quant_config,
            ep_batch_manager=EPBatchManager(ep_config),
            swiglu_limit=swiglu_limit,
        )
    assert moe._blockwise_mega_ffn_amd_eligible()
    module = _LocalExperts(moe)

    rng = np.random.default_rng(29)
    checkpoint: dict[str, Buffer] = {}
    for name, p in module.parameters:
        shape = tuple(int(d) for d in p.shape)
        if p.dtype == f8:
            value = Buffer.from_numpy(_fp8_bytes(rng, shape)).view(f8)
        elif p.dtype == f32:
            lo, hi = (0.005, 0.02) if "down_proj" in name else (0.3, 0.9)
            value = Buffer.from_numpy(
                rng.uniform(lo, hi, shape).astype(np.float32)
            )
        else:
            value = Buffer.from_numpy(
                np.zeros(int(np.prod(shape)) * p.dtype.size_in_bytes, np.uint8)
            ).view(p.dtype, shape)
        checkpoint[name] = value

    device_ref = DeviceRef(device.label, device.id)
    cpu_ref = DeviceRef.CPU()
    input_types = (
        TensorType(f8, (rows, h), device=device_ref),
        TensorType(f32, (h // 128, rows), device=device_ref),
        TensorType(DType.uint32, (n_experts + 1,), device=device_ref),
        TensorType(DType.int32, (n_experts,), device=device_ref),
        TensorType(DType.uint32, (2,), device=cpu_ref),
        TensorType(DType.uint32, (), device=cpu_ref),
    )
    session = InferenceSession(devices=[device])

    def compile_layer() -> tuple[str, CompiledCallable[..., Tensor]]:
        staged, registry = module.stage(*input_types, weights=checkpoint)
        mlir = str(staged.graph)
        return mlir, CompiledCallable(staged, registry, session=session)

    mega_mlir, mega = compile_layer()
    assert "mo.grouped.ffn.blockwise.fp8.amd" in mega_mlir
    monkeypatch.setattr(
        ExpertParallelMoE, "_blockwise_mega_ffn_amd_eligible", lambda _: False
    )
    chain_mlir, chain = compile_layer()
    assert "mo.grouped.ffn.blockwise.fp8.amd" not in chain_mlir

    offsets = np.zeros(n_experts + 1, dtype=np.uint32)
    offsets[1:] = np.cumsum(tokens_per_expert)
    cpu = Accelerator.cpu()

    def arg(a: np.ndarray, dtype: DType | None = None) -> Buffer:
        b = Buffer.from_numpy(a)
        return (b.view(dtype) if dtype else b).to(device)

    args = [
        arg(_fp8_bytes(rng, (rows, h)), f8),
        arg(rng.uniform(0.1, 0.3, (h // 128, rows)).astype(np.float32)),
        arg(offsets),
        arg(np.arange(n_experts, dtype=np.int32)),
        Buffer.from_numpy(
            np.array([max(tokens_per_expert), n_experts], dtype=np.uint32)
        ),
    ]
    for estimated_total_m in (16, total):
        est = Buffer.from_numpy(np.array(estimated_total_m, dtype=np.uint32))
        # Through the engine model rather than the callable, whose device
        # check refuses the host-backed buffers of the precompile record run
        # before the model can say it needs hardware.
        (want_buf,) = chain.engine_model.execute(*args, est)
        (got_buf,) = mega.engine_model.execute(*args, est)
        want = np.from_dlpack(want_buf.to(cpu))[:total]
        got = np.from_dlpack(got_buf.to(cpu))[:total]
        assert np.count_nonzero(want) > want.size // 2
        bad = int(np.count_nonzero(want != got))
        assert bad == 0, (
            f"{bad} of {got.size} outputs differ from the chain at"
            f" estimated_total_m={estimated_total_m}"
        )
