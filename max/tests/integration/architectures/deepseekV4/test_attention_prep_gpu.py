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
"""The fused DeepSeek-V4 attention-prep ops against the graph compositions
they replace, run in the same graph on the same inputs.

The references are the model's own layer code (``weightless_rms_normalize``,
``apply_rope_tail``, ``fp8_qat_quantize``, ``DeepseekV4Compressor``). Every
fused op keeps the graph's rounding points. qkv_prep also keeps its reduction
order, so in bfloat16 it is held to bitwise equality; the graph compressor
pools per-window tables in another order than the in-place kernel, so an
element may land one activation ulp away, and a QAT output one grid step away
when that ulp crosses a rounding midpoint, rarely (:func:`assert_within_ulps`,
:func:`_assert_fp4_close`).

What is exact regardless: which cache slots are written and the raw state
rows.
"""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Sequence
from types import SimpleNamespace
from typing import cast

import ml_dtypes
import numpy as np
import pytest
from max import tree
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn._graph_ops import (
    _dsv4_qkv_prep_ragged,
    compress_windows_ragged,
)
from max.nn.kv_cache import MHAKVCacheParams, MLAKVCacheParams
from max.pipelines.architectures.deepseekV4.layers.attention import (
    weightless_rms_normalize,
)
from max.pipelines.architectures.deepseekV4.layers.compressor import (
    DeepseekV4Compressor,
)
from max.pipelines.architectures.deepseekV4.layers.quantization import (
    fp8_qat_quantize,
)
from max.pipelines.architectures.deepseekV4.layers.rope import (
    apply_rope_tail,
)
from max.pipelines.architectures.deepseekV4.model_config import (
    DeepseekV4Config,
)
from test_common.simple_kv_cache import paged_kv_cache_inputs

pytestmark = pytest.mark.skipif(
    accelerator_count() == 0, reason="the fused ops are exercised on GPU"
)

PAGE = 128
ROPE = 64
MAX_SEQ = 1024
EPS = 1e-6
LAYERS = 2
LAYER = 1
PAGES = 24
BF16 = ml_dtypes.bfloat16

# (tokens already cached, tokens in this chunk) per request.
BATCHES = [
    [(0, 37), (0, 5)],
    [(300, 1), (7, 1), (517, 1)],
    [(128, 64), (2, 9)],
]


@pytest.fixture(scope="module")
def session() -> InferenceSession:
    return InferenceSession(devices=[Accelerator()])


def _to_buffer(a: np.ndarray, dtype: DType) -> Buffer:
    if dtype == DType.bfloat16:
        return Buffer.from_numpy(a.astype(BF16).view(np.uint16)).view(
            DType.bfloat16
        )
    return Buffer.from_numpy(a.astype(np.float32))


def _to_numpy(b: Buffer, dtype: DType) -> np.ndarray:
    host = b.to(CPU())
    if dtype == DType.bfloat16:
        return host.view(DType.uint16).to_numpy().view(BF16).astype(np.float32)
    return host.to_numpy().astype(np.float32)


def _freqs(rng: np.random.Generator, seq: int = MAX_SEQ) -> np.ndarray:
    """``[seq, ROPE // 2, 2]`` unit-magnitude cos/sin."""
    theta = rng.uniform(0, 2 * np.pi, size=(seq, ROPE // 2))
    return np.stack([np.cos(theta), np.sin(theta)], -1).astype(np.float32)


def _ragged(batch: Sequence[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    """``[batch + 1]`` uint32 row offsets and ``[rows]`` int32 positions."""
    offsets = np.zeros(len(batch) + 1, dtype=np.uint32)
    offsets[1:] = np.cumsum([s for _, s in batch])
    rows = int(offsets[-1])
    bid = np.searchsorted(offsets, np.arange(rows), "right") - 1
    pos = np.array([p for p, _ in batch])[bid] + np.arange(rows) - offsets[bid]
    return offsets, pos.astype(np.int32)


def assert_within_ulps(
    got: np.ndarray,
    want: np.ndarray,
    *,
    rel: float,
    max_frac: float = 1.0,
    pairs: bool = False,
) -> None:
    """Every element within ``rel`` of the reference, relative to its own
    magnitude, and at most ``max_frac`` of them different at all.

    With ``pairs`` the scale is the norm of the element's RoPE pair instead:
    a rotated component can cancel to near zero, so its own magnitude is no
    scale for its error, while the rotation keeps the pair's norm.
    """
    assert np.all(np.isfinite(got)), "non-finite output"
    diff = np.abs(got - want)
    if pairs:
        re, im = want[..., 0::2], want[..., 1::2]
        scale = np.repeat(np.sqrt(re * re + im * im), 2, axis=-1)
    else:
        scale = np.abs(want)
    worst = float(np.max(diff / np.maximum(scale, 1e-30), initial=0.0))
    frac = float(np.mean(diff > 0))
    assert worst <= rel, f"worst relative difference {worst} > {rel}"
    assert frac <= max_frac, f"{frac} of elements differ, > {max_frac}"


def _assert_fp4_close(got: np.ndarray, want: np.ndarray) -> None:
    """Each FP4 output finite and at most one e2m1 grid step from the
    reference, and at most 5% of them different.

    Outputs are dequantized e2m1 codes times a power-of-two scale per 32
    columns, and a block's largest code is 3, 4 or 6, so the largest power
    of two at or below a third of the block maximum is the scale (twice it
    when the maximum is a 6, which only loosens the bound). The two sides'
    scales may differ by one power of two, where a block maximum sits on a
    scale boundary; the step is then taken on the coarser grid.
    """
    assert np.all(np.isfinite(got)) and np.all(np.isfinite(want))
    g = got.reshape(-1, 32).astype(np.float64)
    w = want.reshape(-1, 32).astype(np.float64)

    def scale(x: np.ndarray) -> np.ndarray:
        amax = np.maximum(np.max(np.abs(x), -1, keepdims=True), 1e-30)
        return np.exp2(np.floor(np.log2(amax / 3)))

    sg, sw = scale(g), scale(w)
    assert np.all(np.abs(np.log2(sg / sw)) <= 1)
    s = np.maximum(sg, sw)
    mag = np.maximum(np.abs(g), np.abs(w)) / s
    step = np.where(mag < 2, 0.5, np.where(mag < 4, 1.0, 2.0)) * s
    assert np.all(np.abs(g - w) <= step)
    assert float(np.mean(g != w)) <= 0.05


def _leaf_row(
    blocks: np.ndarray, lut: np.ndarray, spp: int, b: int, slot: int, kv: int
) -> np.ndarray:
    return blocks[int(lut[b, slot // spp]), kv, LAYER, slot % spp, 0]


def _leaf_rows(
    blocks: np.ndarray,
    lut: np.ndarray,
    offsets: np.ndarray,
    pos: np.ndarray,
    kv: int = 0,
) -> np.ndarray:
    """Row ``t``'s slot in a token-paged leaf, for every row."""
    bid = np.searchsorted(offsets, np.arange(len(pos)), "right") - 1
    return np.stack(
        [
            _leaf_row(blocks, lut, PAGE, int(bid[t]), int(pos[t]), kv)
            for t in range(len(pos))
        ]
    )


@dataclasses.dataclass
class _QkvModel:
    model: Model
    params: MLAKVCacheParams
    act: DType
    heads: int
    q_lora: int
    seq: int


def _build_qkv(
    session: InferenceSession, act: DType, heads: int, q_lora: int, seq: int
) -> _QkvModel:
    """The fused qkv_prep and its graph composition, in one graph."""
    head_dim = 512
    ref = DeviceRef.GPU()
    params = MLAKVCacheParams(
        dtype=act,
        head_dim=head_dim,
        num_layers=LAYERS,
        devices=[ref],
        page_size=PAGE,
        num_q_heads=heads,
    )
    with Graph(
        "qkv_prep",
        input_types=[
            TensorType(act, ["rows", heads, head_dim], ref),
            TensorType(act, ["rows", q_lora + head_dim], ref),
            TensorType(DType.float32, [head_dim], ref),
            TensorType(DType.float32, [seq, ROPE // 2, 2], ref),
            TensorType(DType.int32, ["rows"], ref),
            TensorType(DType.uint32, ["batch_plus_one"], ref),
            *params.flattened_kv_inputs(),
        ],
    ) as graph:
        q, qkv, gamma, table, pos, offsets = (
            v.tensor for v in graph.inputs[:6]
        )
        collection = params.unflatten_kv_inputs(iter(graph.inputs[6:]))[0]
        fused = _dsv4_qkv_prep_ragged(
            q,
            qkv,
            gamma,
            ops.reshape(table, [seq, ROPE]),
            offsets,
            collection,
            ops.constant(LAYER, DType.uint32, DeviceRef.CPU()),
            kv_col=q_lora,
            epsilon=EPS,
        )
        freqs_cis = ops.gather(table, pos, axis=0)
        q_ref = apply_rope_tail(
            weightless_rms_normalize(ops.unsqueeze(q, 0), EPS), freqs_cis, ROPE
        )
        kv = ops.unsqueeze(qkv[:, q_lora : q_lora + head_dim], 0)
        kv = ops.rms_norm(
            kv,
            gamma.cast(act),
            EPS,
            weight_offset=0.0,
            multiply_before_cast=True,
        )
        kv = apply_rope_tail(kv, freqs_cis, ROPE)
        kv_ref = ops.concat(
            [
                fp8_qat_quantize(kv[..., : head_dim - ROPE]),
                kv[..., head_dim - ROPE :],
            ],
            axis=-1,
        )
        graph.output(fused, ops.squeeze(q_ref, 0), ops.squeeze(kv_ref, 0))
    return _QkvModel(session.load(graph), params, act, heads, q_lora, seq)


def _run_qkv(
    m: _QkvModel,
    batch: Sequence[tuple[int, int]],
    rng: np.random.Generator,
    table: np.ndarray,
    gamma: np.ndarray,
    latent_scale: float,
    pages: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(got_q, want_q, got_kv, want_kv)`` for one batch, as float32."""
    head_dim = 512
    device = Accelerator()
    offsets, pos = _ragged(batch)
    rows = len(pos)
    inputs = paged_kv_cache_inputs(
        m.params,
        [s for _, s in batch],
        cache_lengths=[p for p, _ in batch],
        total_num_pages=pages,
    )
    q_in = rng.standard_normal((rows, m.heads, head_dim))
    qkv_in = rng.standard_normal((rows, m.q_lora + head_dim)) * latent_scale
    got_q, want_q, want_kv = m.model(
        _to_buffer(q_in, m.act).to(device),
        _to_buffer(qkv_in, m.act).to(device),
        Buffer.from_numpy(gamma).to(device),
        Buffer.from_numpy(table).to(device),
        Buffer.from_numpy(pos).to(device),
        Buffer.from_numpy(offsets).to(device),
        *tree.leaves(inputs),
    )
    blocks = _to_numpy(inputs.kv_blocks, m.act)
    lut = inputs.lookup_table.to(CPU()).to_numpy()
    return (
        _to_numpy(got_q, m.act),
        _to_numpy(want_q, m.act),
        _leaf_rows(blocks, lut, offsets, pos),
        _to_numpy(want_kv, m.act),
    )


# Token counts for the bitwise check: decode, small and large prefill, and
# both sides of 148 and of 3552 rows, where the graph's rms_norm_gpu kernels
# (not used by mo.reduce.rms_norm) change layout.
BITWISE_TOKENS = [1, 3, 4, 52, 148, 156, 198, 253, 3552]
BITWISE_SEQ = 4096
# Latent input scales: a typical one, and one whose mean square (~1e-6) is on
# the order of EPS, so the stored latents depend on how epsilon is added.
LATENT_SCALES = (3.0, 1e-3)
BITWISE_PAGES = 72


def _bitwise_batch(t: int) -> list[tuple[int, int]]:
    """Decode-shaped (one token per request) up to 4 tokens, else two
    prefill chunks, one of them behind a cached prefix."""
    if t <= 4:
        return [(300 + 7 * i, 1) for i in range(t)]
    return [(0, t // 2), (128, t - t // 2)]


@pytest.mark.parametrize("heads", [32, 64])
def test_qkv_prep_bitwise_to_graph(
    session: InferenceSession, heads: int
) -> None:
    """The fused op equals the graph composition bit for bit at the model's
    shapes (TP2 and TP1 head counts, ``q_lora_rank`` 1024), bfloat16.

    The kernel sums every row in the graph's reduction order, rounds the
    query's squares before summing them, rotates with the complex multiply
    of the graph's ``rope_ragged`` kernel, and adds the latent RMSNorm's
    epsilon in float32, so nothing is left to round differently. This pins
    the graph's current kernels and codegen rather than the model's math:
    when a graph-side change breaks it, follow it in the kernel (or relax
    this check to :func:`assert_within_ulps`); it is not a kernel bug. Each
    shape also runs a latent whose mean square is on the order of epsilon,
    where any change in how epsilon enters the norm shows up in the stored
    latents.
    """
    m = _build_qkv(session, DType.bfloat16, heads, 1024, BITWISE_SEQ)
    rng = np.random.default_rng(3)
    table = _freqs(rng, BITWISE_SEQ)
    gamma = rng.uniform(0.5, 1.5, 512).astype(np.float32)
    head_w = 512 - ROPE
    report: list[tuple[int, float, dict[str, int]]] = []
    for t, latent_scale in itertools.product(BITWISE_TOKENS, LATENT_SCALES):
        got_q, want_q, got_kv, want_kv = _run_qkv(
            m, _bitwise_batch(t), rng, table, gamma, latent_scale, BITWISE_PAGES
        )
        bits = [a.view(np.uint32) for a in (got_q, want_q, got_kv, want_kv)]
        diffs = {
            "q": bits[0][..., :head_w] != bits[1][..., :head_w],
            "q_rope": bits[0][..., head_w:] != bits[1][..., head_w:],
            "kv_fp8": bits[2][:, :head_w] != bits[3][:, :head_w],
            "kv_rope": bits[2][:, head_w:] != bits[3][:, head_w:],
        }
        counts = {k: int(np.count_nonzero(v)) for k, v in diffs.items()}
        if any(counts.values()):
            report.append((t, latent_scale, counts))
    assert not report, f"fused != graph: {report}"


@pytest.mark.parametrize("act", [DType.bfloat16, DType.float32])
def test_qkv_prep_matches_graph(session: InferenceSession, act: DType) -> None:
    """Small shapes and ragged batches with cached prefixes, both dtypes.

    In float32 no output rounding absorbs the graph's different reduction
    widths, so most elements differ by a few ulps.
    """
    m = _build_qkv(session, act, heads=8, q_lora=256, seq=MAX_SEQ)
    rng = np.random.default_rng(0)
    table = _freqs(rng)
    head_w = 512 - ROPE
    for batch in BATCHES:
        gamma = rng.uniform(0.5, 1.5, 512).astype(np.float32)
        got_q, want_q, got_kv, want_kv = _run_qkv(
            m, batch, rng, table, gamma, 3.0, PAGES
        )
        # FP8 columns: equal, or one e4m3 step (<= 2**-3 relative) apart.
        assert_within_ulps(
            got_kv[:, :head_w],
            want_kv[:, :head_w],
            rel=2.0**-3 + 1e-6,
            max_frac=0.01,
        )
        if act == DType.bfloat16:
            assert_within_ulps(got_q, want_q, rel=2.0**-7, max_frac=0.02)
            assert_within_ulps(
                got_kv[:, head_w:],
                want_kv[:, head_w:],
                rel=2.0**-7,
                max_frac=0.02,
            )
        else:
            assert_within_ulps(
                got_q[..., :head_w], want_q[..., :head_w], rel=2.0**-20
            )
            assert_within_ulps(
                got_q[..., head_w:],
                want_q[..., head_w:],
                rel=2.0**-20,
                pairs=True,
            )
            assert_within_ulps(
                got_kv[:, head_w:],
                want_kv[:, head_w:],
                rel=2.0**-20,
                pairs=True,
            )


def _window_rows(
    batch: Sequence[tuple[int, int]],
    ratio: int,
    coff: int,
    kv_in: np.ndarray,
    score_in: np.ndarray,
    state_blocks: np.ndarray,
    lut: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """numpy window assembly for the graph compressor: kv, score, present,
    start."""
    width = kv_in.shape[1]
    kv, score, present, start = [], [], [], []
    first = 0
    for b, (p, s) in enumerate(batch):
        base = p // ratio
        for j in range(-(-s // ratio)):
            st = (base + j) * ratio
            k_rows, s_rows, live = [], [], []
            for i in range(coff * ratio):
                q = st - (coff - 1) * ratio + i
                if q < 0:
                    k_rows.append(np.zeros(width, np.float32))
                    s_rows.append(np.zeros(width, np.float32))
                elif q < p:
                    k_rows.append(_leaf_row(state_blocks, lut, PAGE, b, q, 0))
                    s_rows.append(_leaf_row(state_blocks, lut, PAGE, b, q, 1))
                else:
                    r = first + min(q - p, s - 1)
                    k_rows.append(kv_in[r])
                    s_rows.append(score_in[r])
                live.append(q >= 0)
            kv.append(np.stack(k_rows).reshape(coff, ratio, width))
            score.append(np.stack(s_rows).reshape(coff, ratio, width))
            present.append(np.array(live).reshape(coff, ratio))
            start.append(st)
        first += s
    return (
        np.stack(kv).astype(np.float32),
        np.stack(score).astype(np.float32),
        np.stack(present),
        np.array(start, np.int32),
    )


def _window_offsets(batch: Sequence[tuple[int, int]], ratio: int) -> np.ndarray:
    woff = np.zeros(len(batch) + 1, dtype=np.uint32)
    woff[1:] = np.cumsum([-(-s // ratio) for _, s in batch])
    return woff


@pytest.mark.parametrize(
    "ratio,head_dim,rotate",
    [(4, 512, False), (128, 512, False), (4, 128, True)],
)
def test_compress_matches_graph(
    session: InferenceSession, ratio: int, head_dim: int, rotate: bool
) -> None:
    """The compressed stream against the graph compressor, with both stores.

    A closed window matches the graph; a window still open after the chunk is
    zero and leaves its zone slot untouched. The output is sized by the
    shape-derived bound the model uses; the last batch leaves it one window
    above the live count, so the rows past the count must leave the zone leaf
    untouched.
    """
    act = DType.bfloat16
    coff = 2 if ratio == 4 else 1
    proj = coff * head_dim
    device = Accelerator()
    ref = DeviceRef.GPU()
    state_params = MHAKVCacheParams(
        dtype=DType.float32,
        n_kv_heads=1,
        head_dim=proj,
        num_layers=LAYERS,
        devices=[ref],
        page_size=PAGE,
    )
    spp = PAGE // ratio
    zone_params = MLAKVCacheParams(
        dtype=act,
        head_dim=head_dim,
        num_layers=LAYERS,
        devices=[ref],
        page_size=PAGE,
        slots_per_page=spp,
        num_q_heads=8,
    )
    config = cast(
        DeepseekV4Config,
        SimpleNamespace(
            hidden_size=64, qk_rope_head_dim=ROPE, rms_norm_eps=EPS, dtype=act
        ),
    )
    compressor = DeepseekV4Compressor(config, ratio, head_dim, ref, rotate)
    rng = np.random.default_rng(1)
    compressor.load_state_dict(
        {
            "ape": rng.standard_normal((ratio, proj)).astype(np.float32),
            "norm.weight": rng.uniform(0.5, 1.5, head_dim).astype(np.float32),
            "wkv_wgate.weight": _to_buffer(np.zeros((2 * proj, 64)), act),
        }
    )
    n_state = len(state_params.flattened_kv_inputs())
    with Graph(
        "compress",
        input_types=[
            TensorType(DType.float32, ["rows", proj], ref),
            TensorType(DType.float32, ["rows", proj], ref),
            TensorType(DType.float32, [MAX_SEQ, ROPE // 2, 2], ref),
            TensorType(DType.uint32, ["batch_plus_one"], ref),
            TensorType(DType.uint32, ["batch_plus_one"], ref),
            TensorType(DType.int32, ["batch"], ref),
            TensorType(DType.float32, ["windows", coff, ratio, proj], ref),
            TensorType(DType.float32, ["windows", coff, ratio, proj], ref),
            TensorType(DType.bool, ["windows", coff, ratio], ref),
            TensorType(DType.int32, ["windows"], ref),
            *state_params.flattened_kv_inputs(),
            *zone_params.flattened_kv_inputs(),
        ],
    ) as graph:
        kv_v, score_v, table, offsets, woff, starts = (
            v.tensor for v in graph.inputs[:6]
        )
        kv_rows, score_rows, present, start = (
            v.tensor for v in graph.inputs[6:10]
        )
        rest = list(graph.inputs[10:])
        state = state_params.unflatten_kv_inputs(iter(rest[:n_state]))[0]
        zone = zone_params.unflatten_kv_inputs(iter(rest[n_state:]))[0]
        fused = compress_windows_ragged(
            kv_v,
            score_v,
            compressor.ape,
            compressor.norm.weight.cast(act).to(ref),
            table,
            offsets,
            woff,
            starts,
            state,
            zone,
            ops.constant(LAYER, DType.uint32, DeviceRef.CPU()),
            epsilon=EPS,
            rotate=rotate,
        )
        want_v = compressor(
            kv_rows,
            score_rows,
            present,
            ops.unsqueeze(ops.gather(table, start, axis=0), 1),
        )
        graph.output(fused, ops.reshape(want_v, [kv_rows.shape[0], head_dim]))
    model = session.load(graph, weights_registry=compressor.state_dict())

    table_np = _freqs(rng)
    for batch in [*BATCHES, [(0, 128), (128, 128)]]:
        offsets_np, pos = _ragged(batch)
        rows = len(pos)
        cached = [p for p, _ in batch]
        chunk = [s for _, s in batch]
        state_in = paged_kv_cache_inputs(
            state_params, chunk, cache_lengths=cached, total_num_pages=PAGES
        )
        zone_in = paged_kv_cache_inputs(
            zone_params, chunk, cache_lengths=cached, total_num_pages=PAGES
        )
        state_blocks = rng.standard_normal(state_in.kv_blocks.shape).astype(
            np.float32
        )
        state_in = dataclasses.replace(
            state_in, kv_blocks=Buffer.from_numpy(state_blocks).to(device)
        )
        zone_before = rng.standard_normal(zone_in.kv_blocks.shape)
        zone_before = zone_before.astype(BF16).astype(np.float32)
        zone_in = dataclasses.replace(
            zone_in, kv_blocks=_to_buffer(zone_before, act).to(device)
        )
        lut = state_in.lookup_table.to(CPU()).to_numpy()
        kv_in = rng.standard_normal((rows, proj)).astype(np.float32)
        score_in = rng.standard_normal((rows, proj)).astype(np.float32)
        kv_np, score_np, present_np, start_np = _window_rows(
            batch, ratio, coff, kv_in, score_in, state_blocks, lut
        )
        got, want = model(
            Buffer.from_numpy(kv_in).to(device),
            Buffer.from_numpy(score_in).to(device),
            Buffer.from_numpy(table_np).to(device),
            Buffer.from_numpy(offsets_np).to(device),
            Buffer.from_numpy(_window_offsets(batch, ratio)).to(device),
            Buffer.from_numpy(np.array(cached, np.int32)).to(device),
            Buffer.from_numpy(kv_np).to(device),
            Buffer.from_numpy(score_np).to(device),
            Buffer.from_numpy(present_np).to(device),
            Buffer.from_numpy(start_np).to(device),
            *tree.leaves(state_in),
            *tree.leaves(zone_in),
        )
        live = len(start_np)
        assert got.shape[0] == (rows + len(batch) * (ratio - 1)) // ratio
        got_np = _to_numpy(got, act)[:live]
        want_np = _to_numpy(want, act)
        closed = np.array(
            [
                (p // ratio + j + 1) * ratio <= p + s
                for p, s in batch
                for j in range(-(-s // ratio))
            ]
        )
        np.testing.assert_array_equal(got_np[~closed], 0)
        got_np, want_np = got_np[closed], want_np[closed]
        # Only the RoPE tail is QAT-free on the FP8 path; the rotated path
        # quantizes everything to e2m1.
        if not closed.any():
            pass
        elif rotate:
            _assert_fp4_close(got_np, want_np)
        else:
            assert_within_ulps(
                got_np, want_np, rel=2.0**-3 + 1e-6, max_frac=0.02
            )

        # The chunk's raw rows landed in the state leaf, bit for bit.
        new_state = _to_numpy(state_in.kv_blocks, DType.float32)
        np.testing.assert_array_equal(
            _leaf_rows(new_state, lut, offsets_np, pos, kv=0), kv_in
        )
        np.testing.assert_array_equal(
            _leaf_rows(new_state, lut, offsets_np, pos, kv=1), score_in
        )
        # Every closed window's entry landed on entry slot ``P // ratio + j``,
        # and nothing else in the zone leaf moved.
        zlut = zone_in.lookup_table.to(CPU()).to_numpy()
        zone_want = zone_before.copy()
        w = 0
        for b, (p, s) in enumerate(batch):
            for j in range(-(-s // ratio)):
                slot = p // ratio + j
                if (slot + 1) * ratio <= p + s:
                    page = int(zlut[b, slot // spp])
                    zone_want[page, 0, LAYER, slot % spp, 0] = got_np[w]
                    w += 1
        np.testing.assert_array_equal(
            _to_numpy(zone_in.kv_blocks, act), zone_want
        )
