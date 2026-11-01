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
"""Tests the trunk's MXFP4 routed experts against the reference's W4A8 math.

The reference (``inference/model.py`` and ``inference/kernel.py`` of the
DeepSeek-V4.1 release) quantizes each expert GEMM's input with
``act_quant(x, 32, "ue8m0")`` -- E4M3 values and one power-of-two scale per
32, the block's absolute max over 448 rounded up -- and multiplies it with
``fp4_gemm`` against E2M1 weights carrying one E8M0 scale per 32. The routing
weight scales the SwiGLU output before the down projection's quantize.

Both paths run here on random experts, with weights converted by the weight
adapter: the routed path (``_routed`` over the ``_grouped`` GEMMs) and the
expert-parallel path (``dispatched`` over rows already sorted by expert,
padded like a receive buffer). The routed path must also match DSpark's
experts, which read the same weights with in-graph interleaved scales, bit
for bit. The expert-parallel path also runs with gate/up rows interleaved,
as the trunk loads them under data parallelism: at small shapes that runs
the fused SwiGLU grouped matmul and the grouped matmul, and at the
DeepSeek-V4.1 Flash expert shapes the graph compiler fuses the two into the
W4A8 MegaFFN. A planted fault (the weights' nibble order swapped in the
reference) must fail the comparison.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops
from max.graph.weights import WeightData
from max.pipelines.architectures.deepseekV41.layers.moe import (
    DeepseekV41Mxfp4RoutedExperts,
    mxfp8_grouped_quantize,
)
from max.pipelines.architectures.deepseekV41.model_config import (
    RoutedExpertFormat,
)
from max.pipelines.architectures.deepseekV41.weight_adapters import (
    as_uint8,
    convert_weight_data,
    with_dtype,
)

HIDDEN = 256
INTER = 256
EXPERTS = 8
TOPK = 2
TOKENS = 37
# Small enough that both sides of the SwiGLU clamp are exercised.
SWIGLU_LIMIT = 2.0
# Rows of the expert-parallel test per local expert; one is empty.
ROWS_PER_EXPERT = [5, 0, 37, 1, 130, 9, 64, 3]
PAD_ROWS = 50

# The DeepSeek-V4.1 Flash expert shapes the fused W4A8 kernel is built for.
FLASH_HIDDEN = 5120
FLASH_INTER = 2304
FLASH_EXPERTS = 4
FLASH_ROWS_PER_EXPERT = [5, 0, 37, 13]
# The fusion sizes the kernel's counters from the receive buffer's rows; a
# buffer this large holds every counter the fused launch needs.
FLASH_RECEIVE_ROWS = 32_768

# Both paths match the reference bit for bit at these sizes. The tolerance
# admits another GEMM accumulation order, which can move a bf16 rounding and
# so an E4M3 code of the down projection's input by one step.
REL_TOL = 2e-2

_E2M1 = np.array(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
    dtype=np.float64,
)


def _checkpoint(
    prefix: str,
    rng: np.random.Generator,
    *,
    experts: int = EXPERTS,
    hidden: int = HIDDEN,
    inter: int = INTER,
) -> dict[str, WeightData]:
    """Per-expert MXFP4 tensors as stored: int8 E2M1 pairs, E8M0 per 32."""
    sd: dict[str, WeightData] = {}
    for e in range(experts):
        for proj, n, k in (
            ("w1", inter, hidden),
            ("w3", inter, hidden),
            ("w2", hidden, inter),
        ):
            base = f"{prefix}.{e}.{proj}"
            weight = rng.integers(0, 256, (n, k // 2), dtype=np.uint8)
            # 2^-6 to 2^-4: outputs of order one.
            scale = rng.integers(121, 124, (n, k // 32), dtype=np.uint8)
            sd[f"{base}.weight"] = with_dtype(
                weight.view(np.int8), DType.int8, f"{base}.weight"
            )
            sd[f"{base}.scale"] = with_dtype(
                scale, DType.float8_e8m0fnu, f"{base}.scale"
            )
    return sd


def _dequantized(
    sd: dict[str, WeightData],
    prefix: str,
    swap_nibbles: bool = False,
    experts: int = EXPERTS,
) -> tuple[np.ndarray, np.ndarray]:
    """Each expert's ``[w1; w3]`` and ``w2`` as float64 values."""

    def values(base: str) -> np.ndarray:
        packed = as_uint8(sd[f"{base}.weight"].data)
        low, high = packed & 0xF, packed >> 4
        if swap_nibbles:
            low, high = high, low
        out = np.empty((packed.shape[0], packed.shape[1] * 2), np.float64)
        # E2M1 packs the even K element in the low nibble.
        out[:, 0::2] = _E2M1[low]
        out[:, 1::2] = _E2M1[high]
        scale = as_uint8(sd[f"{base}.scale"].data)
        return out * np.exp2(scale.astype(np.float64) - 127).repeat(32, -1)

    gate_up = np.stack(
        [
            np.concatenate(
                [values(f"{prefix}.{e}.w1"), values(f"{prefix}.{e}.w3")]
            )
            for e in range(experts)
        ]
    )
    down = np.stack([values(f"{prefix}.{e}.w2") for e in range(experts)])
    return gate_up, down


def _act_quant(x: torch.Tensor) -> torch.Tensor:
    """The reference's ``act_quant(x, 32, "ue8m0")``, dequantized."""
    blocks = x.float().reshape(*x.shape[:-1], -1, 32)
    amax = blocks.abs().amax(-1, keepdim=True).clamp(min=1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    codes = (blocks / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return (codes.float() * scale).reshape(x.shape)


def _fp4_gemm(x: torch.Tensor, w: np.ndarray) -> torch.Tensor:
    """``fp4_gemm(act_quant(x), w)``, bf16 out."""
    a = _act_quant(x).double()
    return (a @ torch.from_numpy(w).T).to(torch.bfloat16)


def _expert(
    x: torch.Tensor,
    gate_up: np.ndarray,
    down: np.ndarray,
    weight: torch.Tensor | None,
    *,
    exact_down: bool = False,
) -> torch.Tensor:
    """The reference's ``Expert.forward``: rows ``x`` through one expert.

    With ``exact_down``, the down projection reads its input unquantized
    and returns float64.
    """
    gate, up = _fp4_gemm(x, gate_up).float().split(down.shape[1], dim=-1)
    up = up.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
    gate = gate.clamp(max=SWIGLU_LIMIT)
    h = torch.nn.functional.silu(gate) * up
    if weight is not None:
        h = weight.unsqueeze(-1) * h
    if exact_down:
        return h.double() @ torch.from_numpy(down).T
    return _fp4_gemm(h.to(torch.bfloat16), down)


def _reference_routed(
    x: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    gate_up: np.ndarray,
    down: np.ndarray,
) -> torch.Tensor:
    y = torch.zeros(x.shape, dtype=torch.float32)
    for e in range(EXPERTS):
        token, slot = torch.nonzero(indices == e, as_tuple=True)
        if len(token):
            y[token] += _expert(
                x[token], gate_up[e], down[e], weights[token, slot]
            ).float()
    return y.to(torch.bfloat16)


def _rel_err(got: torch.Tensor, want: torch.Tensor) -> float:
    diff = (got.double() - want.double()).norm()
    return float(diff / want.double().norm())


def _experts(
    sd: dict[str, WeightData],
    prefix: str,
    device_ref: DeviceRef,
    *,
    host_interleaved_scales: bool,
    interleave_gate_up: bool = False,
    n_experts: int = EXPERTS,
    hidden: int = HIDDEN,
    inter: int = INTER,
) -> DeepseekV41Mxfp4RoutedExperts:
    """Experts loaded strictly with the weight adapter's output.

    With ``interleave_gate_up``, converted as under data parallelism, which
    interleaves the trunk's gate/up rows for the fused kernels.
    """
    is_dspark = prefix.startswith("mtp.")
    converted = convert_weight_data(
        sd,
        num_routed_experts=n_experts,
        num_dspark_experts=n_experts,
        routed_expert_format=RoutedExpertFormat.MXFP4,
        data_parallel_degree=2 if interleave_gate_up else 1,
    )
    experts = DeepseekV41Mxfp4RoutedExperts(
        n_experts,
        hidden,
        inter,
        TOPK,
        SWIGLU_LIMIT,
        device_ref,
        host_interleaved_scales=host_interleaved_scales,
    )
    experts.gate_up_interleaved = interleave_gate_up
    assert host_interleaved_scales != is_dspark
    experts.load_state_dict(
        {
            name.removeprefix(f"{prefix}."): data
            for name, data in converted.items()
        },
        weight_alignment=1,
        strict=True,
    )
    return experts


def _routed_model(
    session: InferenceSession,
    experts: DeepseekV41Mxfp4RoutedExperts,
    device_ref: DeviceRef,
) -> Model:
    with Graph(
        "mxfp4_routed",
        input_types=[
            TensorType(DType.bfloat16, ("t", HIDDEN), device_ref),
            TensorType(DType.float32, ("t", TOPK), device_ref),
            TensorType(DType.int32, ("t", TOPK), device_ref),
        ],
    ) as graph:
        x, weights, indices = (v.tensor for v in graph.inputs)
        graph.output(experts(x, weights, indices))
    return session.load(graph, weights_registry=experts.state_dict())


def _dispatched_model(
    session: InferenceSession,
    experts: DeepseekV41Mxfp4RoutedExperts,
    device_ref: DeviceRef,
    rows: int,
    estimated_rows: int | None = None,
) -> Model:
    """``dispatched`` over bf16 rows quantized as expert dispatch would."""
    n_experts = experts.n_experts
    with Graph(
        "mxfp4_dispatched",
        input_types=[
            TensorType(DType.bfloat16, (rows, experts.hidden), device_ref),
            TensorType(DType.uint32, (n_experts + 1,), device_ref),
            TensorType(DType.uint32, (n_experts,), device_ref),
            TensorType(DType.int32, (n_experts,), device_ref),
            TensorType(DType.uint32, (2,), DeviceRef.CPU()),
        ],
    ) as graph:
        x, starts, scale_offsets, expert_ids, usage = (
            v.tensor for v in graph.inputs
        )
        quantized, scales = mxfp8_grouped_quantize(
            x, starts, scale_offsets, expert_ids
        )
        estimated = ops.constant(
            np.uint32(rows if estimated_rows is None else estimated_rows),
            DType.uint32,
            DeviceRef.CPU(),
        )
        received: Sequence[TensorValue] = (
            quantized,
            scales,
            starts,
            scale_offsets,
            expert_ids,
            usage,
        )
        graph.output(experts.dispatched(received, estimated))
    return session.load(graph, weights_registry=experts.state_dict())


def _to(device: Accelerator, t: torch.Tensor) -> Buffer:
    return Buffer.from_dlpack(t.contiguous()).to(device)


def test_routed_mxfp4_experts_match_the_reference() -> None:
    """The routed path matches the reference's math and DSpark's experts."""
    device = Accelerator()
    device_ref = DeviceRef(device.label, device.id)
    session = InferenceSession(devices=[device])
    rng = np.random.default_rng(0)
    trunk_sd = _checkpoint("layers.0.ffn.experts", rng)
    # The same tensors under DSpark's names.
    dspark_sd = {
        name.replace("layers.0.", "mtp.0.", 1): with_dtype(
            as_uint8(data.data).copy(),
            data.dtype,
            name.replace("layers.0.", "mtp.0.", 1),
        )
        for name, data in trunk_sd.items()
    }
    trunk = _routed_model(
        session,
        _experts(
            trunk_sd,
            "layers.0.ffn.experts",
            device_ref,
            host_interleaved_scales=True,
        ),
        device_ref,
    )
    dspark = _routed_model(
        session,
        _experts(
            dspark_sd,
            "mtp.0.ffn.experts",
            device_ref,
            host_interleaved_scales=False,
        ),
        device_ref,
    )

    x = torch.from_numpy(rng.standard_normal((TOKENS, HIDDEN))).to(
        torch.bfloat16
    )
    weights = torch.from_numpy(rng.uniform(0.1, 1.5, (TOKENS, TOPK))).float()
    indices = torch.from_numpy(
        np.stack(
            [rng.choice(EXPERTS, TOPK, replace=False) for _ in range(TOKENS)]
        ).astype(np.int32)
    )
    inputs = [_to(device, v) for v in (x, weights, indices)]
    (got_buf,) = trunk.execute(*inputs)
    (dspark_buf,) = dspark.execute(*inputs)
    got = torch.from_dlpack(got_buf).cpu()
    assert torch.equal(
        got.view(torch.int16),
        torch.from_dlpack(dspark_buf).cpu().view(torch.int16),
    )

    gate_up, down = _dequantized(trunk_sd, "layers.0.ffn.experts")
    want = _reference_routed(x, weights, indices, gate_up, down)
    err = _rel_err(got, want)
    print(f"routed MXFP4 experts: relative error {err:.2e} vs the reference")
    assert err < REL_TOL

    fault_gate_up, fault_down = _dequantized(
        trunk_sd, "layers.0.ffn.experts", swap_nibbles=True
    )
    fault = _reference_routed(x, weights, indices, fault_gate_up, fault_down)
    assert _rel_err(got, fault) > 10 * REL_TOL


def _run_dispatched(
    rng: np.random.Generator,
    rows_per_expert: Sequence[int],
    rows: int,
    *,
    interleave_gate_up: bool,
    hidden: int = HIDDEN,
    inter: int = INTER,
    estimated_rows: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
    """Runs ``dispatched`` on random experts and checks it against the
    reference over the received rows; padding past them holds junk.

    Returns:
        The received rows' output and input, their per-expert row starts,
        and the dequantized gate/up and down weights.
    """
    n_experts = len(rows_per_expert)
    device = Accelerator()
    device_ref = DeviceRef(device.label, device.id)
    session = InferenceSession(devices=[device])
    sd = _checkpoint(
        "layers.0.ffn.experts",
        rng,
        experts=n_experts,
        hidden=hidden,
        inter=inter,
    )
    received = sum(rows_per_expert)
    model = _dispatched_model(
        session,
        _experts(
            sd,
            "layers.0.ffn.experts",
            device_ref,
            host_interleaved_scales=True,
            interleave_gate_up=interleave_gate_up,
            n_experts=n_experts,
            hidden=hidden,
            inter=inter,
        ),
        device_ref,
        rows,
        estimated_rows,
    )

    starts = np.zeros(n_experts + 1, np.uint32)
    starts[1:] = np.cumsum(rows_per_expert)
    # Each group's scales start on a 128-row tile of its own.
    tiles = np.zeros(n_experts, np.int64)
    tiles[1:] = np.cumsum([-(-n // 128) for n in rows_per_expert])[:-1]
    scale_offsets = (tiles - starts[:-1] // 128).astype(np.uint32)
    expert_ids = np.arange(n_experts, dtype=np.int32)
    usage = np.array([max(rows_per_expert), n_experts], np.uint32)

    x = torch.from_numpy(rng.standard_normal((rows, hidden))).to(torch.bfloat16)
    # Padding past the received rows holds whatever the buffer held.
    x[received:] = 1e4
    (got_buf,) = model.execute(
        _to(device, x),
        _to(device, torch.from_numpy(starts)),
        _to(device, torch.from_numpy(scale_offsets)),
        _to(device, torch.from_numpy(expert_ids)),
        Buffer.from_dlpack(torch.from_numpy(usage)),
    )
    got = torch.from_dlpack(got_buf).cpu()[:received]

    gate_up, down = _dequantized(sd, "layers.0.ffn.experts", experts=n_experts)
    want = torch.cat(
        [
            _expert(x[starts[e] : starts[e + 1]], gate_up[e], down[e], None)
            for e in range(n_experts)
            if rows_per_expert[e]
        ]
    )
    err = _rel_err(got, want)
    print(
        f"dispatched MXFP4 experts ({hidden}x{inter}, interleaved="
        f"{interleave_gate_up}): relative error {err:.2e} vs the reference"
    )
    assert err < REL_TOL
    assert torch.isfinite(got.float()).all()
    return got, x[:received], starts, gate_up, down


def test_dispatched_mxfp4_experts_match_the_reference() -> None:
    """The expert-parallel path matches the reference over the received rows
    and ignores the receive buffer's padding."""
    rng = np.random.default_rng(1)
    received = sum(ROWS_PER_EXPERT)
    got, x, starts, gate_up, down = _run_dispatched(
        rng,
        ROWS_PER_EXPERT,
        received + PAD_ROWS,
        interleave_gate_up=False,
    )

    # The combine weights the rows after the down projection; the
    # reference weights the down projection's input before its quantize. The
    # two orders round the quantized input differently, and neither is
    # further than the other from the unquantized product.
    weight = torch.from_numpy(rng.uniform(0.1, 1.5, received)).float()

    def per_expert(exact_down: bool) -> torch.Tensor:
        return torch.cat(
            [
                _expert(
                    x[starts[e] : starts[e + 1]],
                    gate_up[e],
                    down[e],
                    weight[starts[e] : starts[e + 1]],
                    exact_down=exact_down,
                )
                for e in range(EXPERTS)
                if ROWS_PER_EXPERT[e]
            ]
        )

    exact = per_expert(exact_down=True)
    reference_err = _rel_err(per_expert(exact_down=False), exact)
    combine_err = _rel_err(
        (got.float() * weight.unsqueeze(-1)).to(torch.bfloat16), exact
    )
    print(
        "down projection's input quantize, relative to the unquantized "
        f"product: {reference_err:.2e} weighted before (reference), "
        f"{combine_err:.2e} weighted after (combine)"
    )
    assert combine_err < 1.25 * reference_err


def test_interleaved_dispatched_mxfp4_experts_match_the_reference() -> None:
    """With gate/up rows interleaved, as under data parallelism, the fused
    SwiGLU grouped matmul and the grouped matmul match the reference."""
    _run_dispatched(
        np.random.default_rng(2),
        ROWS_PER_EXPERT,
        sum(ROWS_PER_EXPERT) + PAD_ROWS,
        interleave_gate_up=True,
    )


def test_fused_mxfp4_experts_match_the_reference() -> None:
    """At DeepSeek-V4.1 Flash's expert shapes, a decode-sized batch in a
    full-sized receive buffer runs the fused W4A8 MegaFFN, which must match
    the reference too."""
    _run_dispatched(
        np.random.default_rng(3),
        FLASH_ROWS_PER_EXPERT,
        FLASH_RECEIVE_ROWS,
        interleave_gate_up=True,
        hidden=FLASH_HIDDEN,
        inter=FLASH_INTER,
        estimated_rows=sum(FLASH_ROWS_PER_EXPERT),
    )
