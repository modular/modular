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
"""``mega_moe.tp`` against the routed-expert chain it replaces, bit for bit.

One graph at a TP8 rank's shapes (128 experts, gate/up ``(1536, 6144)``,
down ``(6144, 768)``, NVFP4, top-4) holds three arms on the same tokens:

- the chain ``TensorParallelMoE._routed_experts`` stages today, as graph ops
  on checkpoint-order weights: ``moe_create_indices``, the gather, the grouped
  quantize, the gate/up grouped matmul, ``fused_silu``, the second quantize,
  the down grouped matmul and ``moe.finalize``;
- ``mega_moe.tp`` on the router's top-k ids and the sigma-permuted gate/up
  bundle, as the flag stages it, which runs its fused launch (routing
  included) up to 256 tokens and its in-op chain past them;
- the same op with ``max_fused_tokens=0``, which runs the chain inside the op
  (the regime a step past the fused launch's band takes).

All three outputs must be byte-identical, at decode, verify, ragged and
c64-sized steps, at the top of the fused band and past it, each run twice so
the fused launch's counters relaunch. The two op arms take their counter
workspaces as inputs, and only the fused launch writes one, so each launch
also shows which regime ran.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import (
    BufferType,
    DeviceRef,
    Dim,
    Graph,
    TensorType,
    TensorValue,
    ops,
)
from max.nn._graph_ops import (
    block_scales_interleave,
    grouped_matmul_block_scaled,
    grouped_quantize_dynamic_block_scaled,
    moe_create_indices,
    moe_finalize,
)
from max.nn._graph_ops.ep import fused_silu
from max.pipelines.architectures.mistral_large4_modulev3.layers.mega_moe_tp import (
    COUNTER_WORDS,
    MAX_FUSED_TOKENS,
    _mega_moe_tp,
)

E = 128
"""Routed experts on every rank."""
D = 768
"""One rank's slice of the expert width: 6144 / 8."""
H = 6144
"""The hidden size."""
TOPK = 4
SFV = 16
"""Elements an NVFP4 scale covers."""


def _rand_e4m3(shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    """Random E4M3 bytes with the NaN encoding masked to +0."""
    a = rng.integers(0, 256, size=shape, dtype=np.uint8)
    a[(a & 0x7F) == 0x7F] = 0
    return a


def _sigma(x: np.ndarray) -> np.ndarray:
    """Gate row i to 2i and up row i to 2i + 1, on axis 1."""
    out = np.empty_like(x)
    out[:, 0::2] = x[:, :D]
    out[:, 1::2] = x[:, D:]
    return out


def _interleave(scales: TensorValue, rows: int, groups: int) -> TensorValue:
    """Per-expert `[rows, groups]` scales in the 6-D tcgen05 layout."""
    return ops.stack(
        [
            block_scales_interleave(s.reshape([rows, groups]))
            for s in ops.split(scales, [1] * E, axis=0)
        ],
        axis=0,
    )


def _build_graph(device: DeviceRef) -> Graph:
    tokens = Dim("tokens")
    input_types: list[TensorType | BufferType] = [
        TensorType(DType.bfloat16, [tokens, H], device=device),
        TensorType(DType.int32, [tokens, TOPK], device=device),
        TensorType(DType.float32, [tokens, TOPK], device=device),
        TensorType(DType.uint8, [E, 2 * D, H // 2], device=device),
        TensorType(DType.float8_e4m3fn, [E, 2 * D, H // SFV], device=device),
        TensorType(DType.uint8, [E, 2 * D, H // 2], device=device),
        TensorType(DType.float8_e4m3fn, [E, 2 * D, H // SFV], device=device),
        TensorType(DType.uint8, [E, H, D // 2], device=device),
        TensorType(DType.float8_e4m3fn, [E, H, D // SFV], device=device),
        TensorType(DType.float32, [E], device=device),
        TensorType(DType.float32, [E], device=device),
        TensorType(DType.float32, [E], device=device),
        TensorType(DType.float32, [E], device=device),
        BufferType(DType.uint32, [COUNTER_WORDS], device=device),
        BufferType(DType.uint32, [COUNTER_WORDS], device=device),
    ]
    with Graph("mega_moe_tp", input_types=input_types) as graph:
        (
            x,
            router_idx,
            router_w,
            w13,
            w13_sf,
            w13p,
            w13p_sf,
            w2,
            w2_sf,
            ws2_13,
            in_13,
            ws2_2,
            in_2,
        ) = (v.tensor for v in graph.inputs[:-2])
        fused_counters, in_op_counters = (v.buffer for v in graph.inputs[-2:])
        teo, starts, rto, ids, _, scales_offset = moe_create_indices(
            ops.reshape(router_idx, [-1]), E, needs_scales_offset=True
        )
        b13 = _interleave(w13_sf, 2 * D, H // SFV)
        b13p = _interleave(w13p_sf, 2 * D, H // SFV)
        b2 = _interleave(w2_sf, H, D // SFV)
        one = ops.constant(1.0, DType.float32, device=device)
        es1 = ws2_13 * in_13
        es2 = ws2_2 * in_2
        insc1 = one / in_13
        insc2 = one / in_2

        # the chain TensorParallelMoE._routed_experts stages without the flag
        gathered = ops.gather(x, ops.cast(teo // TOPK, DType.int32), axis=0)
        usage = ops.shape_to_tensor([gathered.shape[0], ids.shape[0]]).cast(
            DType.uint32
        )
        a, a_sf = grouped_quantize_dynamic_block_scaled(
            gathered,
            row_offsets=starts,
            scales_offsets=scales_offset,
            expert_ids=ids,
            sf_tensor=insc1,
            sf_vector_size=SFV,
            scales_type=DType.float8_e4m3fn,
            out_type=DType.uint8,
        )
        h = grouped_matmul_block_scaled(
            a, w13, a_sf, b13, starts, scales_offset, ids, es1, usage
        )
        y = fused_silu(h, starts)
        o, o_sf = grouped_quantize_dynamic_block_scaled(
            y,
            row_offsets=starts,
            scales_offsets=scales_offset,
            expert_ids=ids,
            sf_tensor=insc2,
            sf_vector_size=SFV,
            scales_type=DType.float8_e4m3fn,
            out_type=DType.uint8,
        )
        down = grouped_matmul_block_scaled(
            o, w2, o_sf, b2, starts, scales_offset, ids, es2, usage
        )
        chain = moe_finalize(down, rto, router_w, DType.bfloat16)

        args = (
            x,
            ops.reshape(router_idx, [-1]),
            router_w,
            w13p,
            b13p,
            w2,
            b2,
            es1,
            es2,
            insc1,
            insc2,
        )
        fused = _mega_moe_tp(*args, counters=fused_counters)
        in_op_chain = _mega_moe_tp(
            *args, max_fused_tokens=0, counters=in_op_counters
        )
        # The control: checkpoint-order W13 where the op reads the permuted
        # one pairs gate rows with gate rows, so it must not match.
        wrong_layout = _mega_moe_tp(*args[:3], w13, b13, *args[5:])
        graph.output(chain, fused, in_op_chain, wrong_layout)
    return graph


@pytest.fixture(scope="module")
def device() -> Accelerator:
    return Accelerator()


@pytest.fixture(scope="module")
def model(device: Accelerator) -> Model:
    session = InferenceSession(devices=[device])
    return session.load(_build_graph(DeviceRef.GPU()))


@pytest.fixture(scope="module")
def counters(device: Accelerator) -> list[Buffer]:
    """The two op arms' counter workspaces, zeroed once for the module, as
    the graph's own are zeroed once at load."""
    return [
        Buffer.from_numpy(np.zeros(COUNTER_WORDS, dtype=np.uint32)).to(device)
        for _ in range(2)
    ]


@pytest.fixture(scope="module")
def weights(device: Accelerator) -> list[Buffer]:
    rng = np.random.default_rng(1234)
    w13 = rng.integers(0, 256, size=(E, 2 * D, H // 2), dtype=np.uint8)
    w13_sf = _rand_e4m3((E, 2 * D, H // SFV), rng)
    w2 = rng.integers(0, 256, size=(E, H, D // 2), dtype=np.uint8)
    w2_sf = _rand_e4m3((E, H, D // SFV), rng)
    # distinct per expert, so a scale read by the wrong index shows
    ws2_13 = rng.uniform(1e-3, 3e-3, E).astype(np.float32)
    in_13 = rng.uniform(0.5, 2.0, E).astype(np.float32)
    ws2_2 = rng.uniform(1e-3, 3e-3, E).astype(np.float32)
    in_2 = rng.uniform(0.5, 2.0, E).astype(np.float32)

    def buf(a: np.ndarray, dtype: DType | None = None) -> Buffer:
        b = Buffer.from_dlpack(torch.from_numpy(np.ascontiguousarray(a)))
        if dtype is not None:
            b = b.view(dtype)
        return b.to(device)

    return [
        buf(w13),
        buf(w13_sf, DType.float8_e4m3fn),
        buf(_sigma(w13)),
        buf(_sigma(w13_sf), DType.float8_e4m3fn),
        buf(w2),
        buf(w2_sf, DType.float8_e4m3fn),
        buf(ws2_13),
        buf(in_13),
        buf(ws2_2),
        buf(in_2),
    ]


def _routing(ntok: int, experts: int, rng: np.random.Generator) -> np.ndarray:
    """Each token's 4 distinct experts, drawn from `experts` spread over 128."""
    pool = (np.arange(experts) * E) // experts
    return np.stack(
        [rng.choice(pool, size=TOPK, replace=False) for _ in range(ntok)]
    ).astype(np.int32)


@pytest.mark.parametrize(
    "ntok,experts",
    [
        (1, 128),  # c1 decode
        (4, 6),  # c1 verify, 6 experts
        (4, 16),
        (17, 8),  # prime tokens, two-pool experts
        (65, 128),
        (162, 128),
        (228, 128),
        (256, 128),  # c64 verify step: the top of the fused band
        (1024, 128),  # past the band: the op runs its chain
    ],
)
def test_fused_op_matches_the_chain(
    model: Model,
    weights: list[Buffer],
    counters: list[Buffer],
    device: Accelerator,
    ntok: int,
    experts: int,
) -> None:
    rng = np.random.default_rng(ntok * 1000 + experts)
    x = torch.from_numpy(
        rng.uniform(-4.0, 4.0, (ntok, H)).astype(np.float32)
    ).to(torch.bfloat16)
    idx = _routing(ntok, experts, rng)
    w = rng.uniform(0.05, 1.0, (ntok, TOPK)).astype(np.float32)
    inputs = [
        Buffer.from_dlpack(x).to(device),
        Buffer.from_dlpack(torch.from_numpy(idx)).to(device),
        Buffer.from_dlpack(torch.from_numpy(w)).to(device),
        *weights,
        *counters,
    ]
    fuses = ntok <= MAX_FUSED_TOKENS
    for launch in range(2):
        before = [c.to_numpy() for c in counters]
        outs = model.execute(*inputs)
        fused_ran, in_op_fused = (
            not np.array_equal(c.to_numpy(), b)
            for c, b in zip(counters, before, strict=True)
        )
        assert fused_ran == fuses, (
            f"launch {launch}: {ntok} tokens took the"
            f" {'fused launch' if fused_ran else 'in-op chain'}, expected the"
            f" {'fused launch' if fuses else 'in-op chain'}"
        )
        assert not in_op_fused, (
            f"launch {launch}: the max_fused_tokens=0 arm wrote its counters,"
            " so it took the fused launch"
        )
        chain, fused, in_op, wrong = (
            torch.from_dlpack(t).cpu().view(torch.int16).numpy() for t in outs
        )
        assert np.count_nonzero(chain) > chain.size // 2, (
            "the chain's output is mostly zero: the gate cannot fail"
        )
        assert np.count_nonzero(wrong != chain) > chain.size // 2, (
            "the op matches the chain on the wrong gate/up layout: the"
            " comparison cannot fail"
        )
        bad_fused = int(np.count_nonzero(fused != chain))
        bad_in_op = int(np.count_nonzero(in_op != chain))
        assert bad_fused == 0, (
            f"launch {launch}: {bad_fused} of {chain.size} fused outputs"
            " differ from the chain"
        )
        assert bad_in_op == 0, (
            f"launch {launch}: {bad_in_op} of {chain.size} in-op chain"
            " outputs differ from the chain"
        )
