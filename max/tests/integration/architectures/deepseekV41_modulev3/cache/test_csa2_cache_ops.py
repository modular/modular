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
"""Runs the registered V2 CSA2 ops over caches the ModuleV3 interface sizes.

The graphs' cache blocks take their shapes from
:func:`deepseekV41_modulev3.interface.build_kv_params`, the stores and the
attention come from the V2 ``layers/csa2`` wrappers unchanged, and the slot
lists are built in-graph by the V2 ``window_entries`` and ``leaf_slots``
helpers over real lookup tables -- so the test fails unless the interface's
geometry is the one the kernels address.

One request runs 288 tokens through three forwards, staged the way the
block manager stages them: every forward's lookup table row maps each live
logical page to a distinct physical block, so no store call ever races.
Forward one writes tokens 0..127 to block 0; forward two writes tokens
128..255 to block 1; forward three writes tokens 256..287 to page 2, which
the manager serves from the recycled block 0 -- the retired page-0 slots
0..31 are overwritten with the newest tokens, the only slot mutation the
ring ever performs. The recycled slots' dequantized value and scale bytes
are checked against the latest tokens directly -- direction and magnitude,
so a wrong uniform store scale cannot pass -- and the un-recycled slots
of the same block against the first forward's tokens.

The ratio-2 compressed leaf stores one NVFP4 record per closed group --
every group of the run closes at an odd position -- across three distinct
blocks, with ten groups skipped by ``-1`` slots whose value and scale
planes must keep their initial bytes; the ratio-1 leaf stores one record
per token. Attention then runs in the third forward as a decode and a
128-row prefill, over lists with a nonzero SWA begin floor, a row whose
compressed selection is empty, and rows whose whole window lies below the
floor. A float64 oracle over the dequantized records checks every decode
row and every prefill row: the prefill spans the first forward only, so
none of its keys depend on recycled state.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import (
    BufferType,
    DeviceRef,
    Graph,
    TensorType,
    Type,
    ops,
)
from max.nn.kv_cache import KVCacheParams, PagedCacheValues
from max.pipelines.architectures.deepseekV41.layers.csa2 import (
    HEAD_DIM,
    MAX_COMPRESSED,
    csa2_attention,
    leaf_slots,
    store_compressed_kv,
    store_window_kv,
    window_entries,
)
from max.pipelines.architectures.deepseekV41_modulev3.interface import (
    PAGE_SIZE,
    V41Config,
    build_kv_params,
)
from max.pipelines.lib import KVCacheConfig

NH = 8
SCALE = HEAD_DIM**-0.5
T = 288  # the request's tokens: three forwards
N_HEAD = PAGE_SIZE  # tokens the first forward stores (block 0)
N_MID = PAGE_SIZE  # tokens the second forward stores (block 1)
N_TAIL = T - N_HEAD - N_MID  # tokens the third forward stores (recycled)
N_PRE = PAGE_SIZE  # prefill rows: the first forward's queries
N_GROUPS = T // 2  # ratio-2 groups, each closing at an odd position
LAYER = 2  # a nonzero layer of the shared window leaf
VALID_FROM = 32  # a nonzero SWA begin floor
SKIP_LO, SKIP_HI = 10, 20  # groups whose stores pass ``-1`` slots
DECODE_POS = [260, 280, 284, 285, 286, 287]
EMPTY_ROW = 1  # the decode row whose compressed selection is empty
ROPE_ROWS = 4096
W_DATA, C_DATA = HEAD_DIM, HEAD_DIM // 2
STORE_COSINE_FLOOR = 0.995  # the MXFP8 store precision the kernel shows
# The staged lookup tables: each forward's row maps every live logical page
# to a distinct physical block. The third forward's page 2 is served from
# block 0, retired after the window advanced past tokens 0..127 and then
# recycled -- the only slot mutation the ring performs.
LUT_HEAD = [[0, 0, 0]]
LUT_MID = [[0, 1, 0]]
LUT_TAIL = [[0, 1, 0]]


def _hf_config() -> dict[str, Any]:
    return {
        "text_config": {
            "vocab_size": 128,
            "num_hidden_layers": 4,
            "num_attention_heads": NH,
            "head_dim": HEAD_DIM,
            "sliding_window": 128,
            "compress_ratios": [0, 2, 1, 0],
            "kv_source_layer_ids": [1, 2],
            "index_head_dim": 128,
            "engram_num_embeddings": [2],
            "engram_head_dim": 256,
            "dspark_target_layer_ids": [],
        }
    }


def _leaf_shapes() -> dict[str, tuple[int, ...]]:
    """The interface's per-block shapes: ``[1, layers, slots, 1, width]``."""
    config = V41Config.from_hf(_hf_config())
    params = build_kv_params(
        config,
        KVCacheConfig(kv_cache_page_size=PAGE_SIZE),
        [DeviceRef.GPU()],
    )
    shapes = {}
    for key, child in params.children.items():
        assert isinstance(child, KVCacheParams), key
        shapes[key] = tuple(int(d) for d in child.shape_per_block)
    return shapes


def bf16_round(x: np.ndarray) -> np.ndarray:
    """Rounds float32 values to the nearest bf16, ties to even."""
    b = np.ascontiguousarray(x, np.float32).view(np.uint32).astype(np.uint64)
    r = (b + 0x7FFF + ((b >> 16) & 1)) >> 16 << 16
    return r.astype(np.uint32).view(np.float32)


def _e4m3_table() -> np.ndarray:
    b = np.arange(256)
    e, m = (b >> 3) & 15, (b & 7).astype(np.float64)
    v = np.where(e == 0, m / 8 * 2.0**-6, (1 + m / 8) * 2.0 ** (e - 7))
    v[(e == 15) & ((b & 7) == 7)] = np.nan
    return np.where(b & 0x80, -v, v)


E4M3 = _e4m3_table()
E2M1 = np.array(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]
)


def _planes(page: np.ndarray, data: int) -> tuple[np.ndarray, np.ndarray]:
    """Splits one layer's ``[slots, record_bytes]`` page into its value and
    scale planes: a page holds every record's value bytes, then every
    record's scale bytes."""
    flat = page.reshape(-1)
    slots = page.shape[0]
    values = flat[: slots * data].reshape(slots, data)
    scales = flat[slots * data :].reshape(slots, -1)
    return values, scales


def dequant_page(page: np.ndarray, data: int) -> np.ndarray:
    """Dequantizes one layer's page to ``[slots, 512]`` float64."""
    values, scales = _planes(page, data)
    if data == W_DATA:
        return E4M3[values] * np.repeat(
            2.0 ** (scales.astype(np.float64) - 127), 32, 1
        )
    vals = np.stack([E2M1[values & 15], E2M1[values >> 4]], -1)
    return vals.reshape(-1, HEAD_DIM) * np.repeat(E4M3[scales], 16, 1)


def rope_table() -> np.ndarray:
    freq = 10000.0 ** (-np.arange(32) / 32)
    angle = np.arange(ROPE_ROWS)[:, None] * freq[None, :]
    return np.concatenate([np.cos(angle), np.sin(angle)], 1).astype(np.float32)


def rope(x: np.ndarray, pos: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Rotates channel pairs 448 to 511 of ``x`` at ``pos``, as the store does."""
    out = x.astype(np.float64)
    cos = table[pos, :32].astype(np.float64)
    sin = table[pos, 32:].astype(np.float64)
    a, b = out[:, 448::2].copy(), out[:, 449::2].copy()
    out[:, 448::2] = a * cos - b * sin
    out[:, 449::2] = a * sin + b * cos
    return out


def _selections() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The compressed selections, as closing positions padded with ``-1``.

    Row ``t`` of the prefill names every group that has closed at or before
    position ``t``; the decode rows name every stored group that has closed
    at or before their position, except row ``EMPTY_ROW``, which names
    none.
    """
    groups = np.arange(N_GROUPS)
    stored = groups[(groups < SKIP_LO) | (groups >= SKIP_HI)]

    def row(max_pos: int, *, empty: bool = False) -> np.ndarray:
        sel = np.full(MAX_COMPRESSED, -1, np.int64)
        if not empty:
            closing = 2 * stored + 1
            sel[: np.count_nonzero(closing <= max_pos)] = closing[
                closing <= max_pos
            ]
        return sel

    prefill = np.stack([row(t) for t in range(N_PRE)])
    decode = np.stack(
        [row(p, empty=(i == EMPTY_ROW)) for i, p in enumerate(DECODE_POS)]
    )
    # Ratio 1: every token closes its own group, at its own position.
    decode1 = np.full((len(DECODE_POS), MAX_COMPRESSED), -1, np.int64)
    for i, p in enumerate(DECODE_POS):
        decode1[i, : p + 1] = np.arange(p + 1)
    return prefill, decode, decode1


def _leaf(
    blocks: Any,
    table: list[list[int]],
    lengths: int,
    gpu: DeviceRef,
    cpu: DeviceRef,
) -> PagedCacheValues:
    return PagedCacheValues(
        kv_blocks=blocks,
        cache_lengths=ops.constant([lengths], DType.int64, gpu),
        lookup_table=ops.constant(table, DType.int64, gpu),
        max_prompt_length=ops.constant(T, DType.int64, cpu),
        max_cache_length=ops.constant(T, DType.int64, cpu),
        page_stride=ops.constant([-1], DType.int64, cpu),
    )


def build_window_store(
    shapes: dict[str, tuple[int, ...]], rows: int, lut: list[list[int]]
) -> Graph:
    """One forward's window store: ``rows`` rows through ``lut``.

    Each forward stages its own lookup table row, the way the block manager
    stages one before the forward; the rows of one forward name distinct
    slots, so the store cannot race.
    """
    gpu, cpu = DeviceRef.GPU(), DeviceRef.CPU()
    swa_shape = (2, *shapes["swa"])
    assert swa_shape[3] == PAGE_SIZE, swa_shape
    inputs: list[Type[Any]] = [
        TensorType(DType.float32, [rows, HEAD_DIM], gpu),
        TensorType(DType.int64, [rows], gpu),
        TensorType(DType.float32, [ROPE_ROWS, 64], gpu),
        BufferType(DType.uint8, list(swa_shape), gpu),
    ]
    with Graph(
        f"csa2_modulev3_store_{rows}_{''.join(str(c) for c in lut[0])}",
        input_types=inputs,
    ) as g:
        latent, pos, table = (v.tensor for v in g.inputs[:3])
        swa_blocks = g.inputs[3].buffer
        swa = _leaf(swa_blocks, lut, T, gpu, cpu)
        bid = ops.broadcast_to(ops.constant(0, DType.int32, gpu), [rows])
        slots = leaf_slots(swa, bid, ops.cast(pos, DType.int32), LAYER)
        store_window_kv(
            swa_blocks,
            latent,
            pos,
            table,
            slots,
            block_stride=swa.page_stride,
        )
        g.output(slots)
    return g


def build_store_tail(shapes: dict[str, tuple[int, ...]]) -> Graph:
    """The third forward: the recycled window store, both compressed
    stores, and the attention, all through the V2 wrappers."""
    gpu, cpu = DeviceRef.GPU(), DeviceRef.CPU()
    swa_shape = (2, *shapes["swa"])
    c2_shape = (3, *shapes["c2a"])
    c1_shape = (3, *shapes["c1a"])
    assert c2_shape[3] == PAGE_SIZE // 2, c2_shape
    assert c1_shape[3] == PAGE_SIZE, c1_shape
    n_dec = len(DECODE_POS)
    inputs: list[Type[Any]] = [
        TensorType(DType.float32, [N_TAIL, HEAD_DIM], gpu),
        TensorType(DType.int64, [N_TAIL], gpu),
        TensorType(DType.float32, [N_GROUPS, HEAD_DIM], gpu),
        TensorType(DType.int64, [N_GROUPS], gpu),
        TensorType(DType.float32, [T, HEAD_DIM], gpu),
        TensorType(DType.int64, [T], gpu),
        TensorType(DType.float32, [ROPE_ROWS, 64], gpu),
        TensorType(DType.float32, [n_dec, NH, HEAD_DIM], gpu),
        TensorType(DType.float32, [N_PRE, NH, HEAD_DIM], gpu),
        TensorType(DType.float32, [NH], gpu),
        TensorType(DType.int32, [1], gpu),
        TensorType(DType.int32, [n_dec, MAX_COMPRESSED], gpu),
        TensorType(DType.int32, [N_PRE, MAX_COMPRESSED], gpu),
        TensorType(DType.int32, [n_dec, MAX_COMPRESSED], gpu),
        BufferType(DType.uint8, list(swa_shape), gpu),
        BufferType(DType.uint8, list(c2_shape), gpu),
        BufferType(DType.uint8, list(c1_shape), gpu),
    ]
    with Graph("csa2_modulev3_store_tail", input_types=inputs) as g:
        (
            w_latent,
            w_pos,
            c2_latent,
            c2_pos,
            c1_latent,
            c1_pos,
            rope,
            q_dec,
            q_pre,
            sink,
            valid_from,
            sel_dec2,
            sel_pre2,
            sel_dec1,
        ) = (v.tensor for v in g.inputs[:14])
        swa_blocks, c2_blocks, c1_blocks = (v.buffer for v in g.inputs[14:])

        # The window ring: this forward's page 2 is served from block 0,
        # retired and recycled, so its rows overwrite the retired page-0
        # slots 0..31 with the newest tokens.
        swa = _leaf(swa_blocks, LUT_TAIL, T, gpu, cpu)
        # The compressed leaves keep every group: three distinct blocks.
        c2 = _leaf(c2_blocks, [[0, 1, 2]], T, gpu, cpu)
        c1 = _leaf(c1_blocks, [[0, 1, 2]], T, gpu, cpu)
        bid_tail = ops.broadcast_to(ops.constant(0, DType.int32, gpu), [N_TAIL])
        store_window_kv(
            swa_blocks,
            w_latent,
            w_pos,
            rope,
            leaf_slots(swa, bid_tail, ops.cast(w_pos, DType.int32), LAYER),
            block_stride=swa.page_stride,
        )
        # Ratio 2: one record per closed group; the skipped groups pass
        # ``-1`` and must leave their slots untouched. Every group of the
        # run closes at an odd position.
        groups = ops.range(
            0, N_GROUPS, 1, out_dim=N_GROUPS, dtype=DType.int32, device=gpu
        )
        skip = (groups >= SKIP_LO) & (groups < SKIP_HI)
        bid_g = ops.broadcast_to(ops.constant(0, DType.int32, gpu), [N_GROUPS])
        store_compressed_kv(
            c2_blocks,
            c2_latent,
            c2_pos,
            rope,
            ops.where(
                skip,
                ops.constant(-1, DType.int32, gpu),
                leaf_slots(c2, bid_g, groups, 0),
            ),
            ratio=2,
            block_stride=c2.page_stride,
        )
        # Ratio 1: one record per token.
        bid_t = ops.broadcast_to(ops.constant(0, DType.int32, gpu), [T])
        store_compressed_kv(
            c1_blocks,
            c1_latent,
            c1_pos,
            rope,
            leaf_slots(c1, bid_t, ops.cast(c1_pos, DType.int32), 0),
            ratio=1,
            block_stride=c1.page_stride,
        )

        def index_list(
            positions: Any, sel: Any, comp: PagedCacheValues, ratio: int
        ) -> Any:
            t = int(positions.shape[0])
            # The V2 ``_index_list`` contract: ``window_entries`` takes
            # the request ids 1-D (it gathers each row's floor), while
            # ``leaf_slots`` takes them row-shaped, [t, 1], to broadcast
            # against the [t, entries] lists.
            bid = ops.broadcast_to(ops.constant(0, DType.int32, gpu), [t])
            request = ops.unsqueeze(bid, -1)
            window = leaf_slots(
                swa,
                request,
                window_entries(positions, bid, valid_from),
                LAYER,
            )
            if ratio == 1:
                group = sel
            else:
                # Closing position -> group, divided exactly once.
                group = ops.where(
                    sel >= 0,
                    ops.floor_div(sel, ratio),
                    ops.constant(-1, DType.int32, gpu),
                )
            return ops.concat([window, leaf_slots(comp, request, group, 0)], 1)

        def attend(
            q: Any,
            idx: Any,
            comp: Any,
            comp_stride: Any,
            *,
            prefill: bool,
        ) -> Any:
            return ops.cast(
                csa2_attention(
                    ops.cast(q, DType.bfloat16),
                    idx,
                    sink,
                    swa_blocks,
                    comp,
                    scale=SCALE,
                    window_stride=swa.page_stride,
                    comp_stride=comp_stride,
                    prefill=prefill,
                ),
                DType.float32,
            )

        dec_pos = ops.constant(DECODE_POS, DType.int32, gpu)
        pre_pos = ops.constant(list(range(N_PRE)), DType.int64, gpu)
        g.output(
            attend(
                q_dec,
                index_list(dec_pos, sel_dec2, c2, 2),
                c2_blocks,
                c2.page_stride,
                prefill=False,
            ),
            attend(
                q_pre,
                index_list(pre_pos, sel_pre2, c2, 2),
                c2_blocks,
                c2.page_stride,
                prefill=True,
            ),
            attend(
                q_dec,
                index_list(dec_pos, sel_dec1, c1, 1),
                c1_blocks,
                c1.page_stride,
                prefill=False,
            ),
        )
    return g


@dataclass
class Run:
    caches: dict[str, np.ndarray]
    initial: dict[str, np.ndarray]
    dec2: np.ndarray
    pre2: np.ndarray
    dec1: np.ndarray
    inputs: dict[str, np.ndarray]


@pytest.fixture(scope="module")
def run() -> Iterator[Run]:
    shapes = _leaf_shapes()
    session = InferenceSession(devices=[CPU(), Accelerator(0)])
    head = session.load(build_window_store(shapes, N_HEAD, LUT_HEAD))
    mid = session.load(build_window_store(shapes, N_MID, LUT_MID))
    tail = session.load(build_store_tail(shapes))
    rng = np.random.default_rng(41)
    w_latent = bf16_round(rng.normal(size=(T, HEAD_DIM)).astype(np.float32))
    c2_latent = bf16_round(
        rng.normal(size=(N_GROUPS, HEAD_DIM)).astype(np.float32)
    )
    c1_latent = bf16_round(rng.normal(size=(T, HEAD_DIM)).astype(np.float32))
    q_decode = bf16_round(
        rng.normal(size=(len(DECODE_POS), NH, HEAD_DIM)).astype(np.float32)
    )
    q_prefill = bf16_round(
        rng.normal(size=(N_PRE, NH, HEAD_DIM)).astype(np.float32)
    )
    sink = rng.normal(size=NH).astype(np.float32)
    pre_sel, dec_sel, dec1_sel = _selections()
    # unwritten bytes are random, NaN codes included
    caches = {
        "swa": rng.integers(0, 256, size=(2, *shapes["swa"]), dtype=np.uint8),
        "c2a": rng.integers(0, 256, size=(3, *shapes["c2a"]), dtype=np.uint8),
        "c1a": rng.integers(0, 256, size=(3, *shapes["c1a"]), dtype=np.uint8),
    }
    table = rope_table()
    w_pos = np.arange(T, dtype=np.int64)
    gpu = Accelerator(0)
    swa_buf, c2_buf, c1_buf = [
        Buffer.from_numpy(np.ascontiguousarray(caches[name])).to(gpu)
        for name in ("swa", "c2a", "c1a")
    ]

    def store(model: Any, rows: slice) -> None:
        model.execute(
            Buffer.from_numpy(np.ascontiguousarray(w_latent[rows])).to(gpu),
            Buffer.from_numpy(np.ascontiguousarray(w_pos[rows])).to(gpu),
            Buffer.from_numpy(np.ascontiguousarray(table)).to(gpu),
            swa_buf,
        )

    # Forward one, then two: distinct blocks, host-sequenced.
    store(head, slice(0, N_HEAD))
    store(mid, slice(N_HEAD, N_HEAD + N_MID))
    tail_buffers = [
        Buffer.from_numpy(np.ascontiguousarray(w_latent[N_HEAD + N_MID :])).to(
            gpu
        ),
        Buffer.from_numpy(np.ascontiguousarray(w_pos[N_HEAD + N_MID :])).to(
            gpu
        ),
        Buffer.from_numpy(np.ascontiguousarray(c2_latent)).to(gpu),
        Buffer.from_numpy(
            np.ascontiguousarray((2 * np.arange(N_GROUPS) + 1).astype(np.int64))
        ).to(gpu),
        Buffer.from_numpy(np.ascontiguousarray(c1_latent)).to(gpu),
        Buffer.from_numpy(np.ascontiguousarray(w_pos)).to(gpu),
        Buffer.from_numpy(np.ascontiguousarray(table)).to(gpu),
        Buffer.from_numpy(np.ascontiguousarray(q_decode)).to(gpu),
        Buffer.from_numpy(np.ascontiguousarray(q_prefill)).to(gpu),
        Buffer.from_numpy(np.ascontiguousarray(sink)).to(gpu),
        Buffer.from_numpy(
            np.ascontiguousarray(np.array([VALID_FROM], np.int32))
        ).to(gpu),
        Buffer.from_numpy(np.ascontiguousarray(dec_sel.astype(np.int32))).to(
            gpu
        ),
        Buffer.from_numpy(np.ascontiguousarray(pre_sel.astype(np.int32))).to(
            gpu
        ),
        Buffer.from_numpy(np.ascontiguousarray(dec1_sel.astype(np.int32))).to(
            gpu
        ),
        swa_buf,
        c2_buf,
        c1_buf,
    ]
    outs = tail.execute(*tail_buffers)
    yield Run(
        caches={
            "swa": swa_buf.to_numpy(),
            "c2a": c2_buf.to_numpy(),
            "c1a": c1_buf.to_numpy(),
        },
        initial=caches,
        dec2=outs[0].to_numpy(),
        pre2=outs[1].to_numpy(),
        dec1=outs[2].to_numpy(),
        inputs={
            "w_latent": w_latent,
            "q_decode": q_decode,
            "q_prefill": q_prefill,
            "sink": sink,
            "sel_dec2": dec_sel,
            "sel_pre2": pre_sel,
            "sel_dec1": dec1_sel,
        },
    )


def _records(cache: np.ndarray, phys: int, layer: int) -> np.ndarray:
    """``[slots, record_bytes]`` bytes of one physical page at one layer."""
    assert cache.shape[1] == 1
    pages, _, layers, slots, _, pitch = (int(d) for d in cache.shape)
    return cache.reshape(pages, layers, slots, pitch)[phys, layer]


def _cosine(got: np.ndarray, want: np.ndarray) -> np.ndarray:
    return (got * want).sum(1) / np.sqrt(
        (got * got).sum(1) * (want * want).sum(1)
    )


def _check_records(got: np.ndarray, want: np.ndarray, what: str) -> None:
    """Checks stored records against their expected rows in magnitude as
    well as direction.

    The cosine alone cannot catch a wrong uniform scale: ``2 * want`` has
    cosine 1. The pinned tree has no Python store reference (the
    byte-exact reference is ``Kernels/test/csa/test_kv_store.mojo``), so
    the bound comes from the MXFP8 format the store writes: one
    power-of-two UE8M0 scale per 32 channels and E4M3 values with a
    3-bit mantissa, so each element carries at most a 2**-4 relative
    error and each 32-channel group's dequantized L2 norm stays within
    that of the expected row (the f32 RoPE arithmetic adds ~1e-7
    relative). The 10% bound holds with a 1.6x margin; a uniform scale
    bug multiplies every group's ratio by the same factor, so 2x or 0.5x
    fails this check while every cosine stays 1.
    """
    cos = _cosine(got, want)
    assert cos.min() > STORE_COSINE_FLOOR, (
        f"{what}: stored records point the wrong way, cosine {cos.min():.4f}"
    )
    groups = got.reshape(len(got), -1, 32)
    expected = want.reshape(len(want), -1, 32)
    ratio = np.linalg.norm(groups, axis=-1) / np.linalg.norm(expected, axis=-1)
    assert np.all((ratio > 0.9) & (ratio < 1.1)), (
        f"{what}: group norm ratios [{ratio.min():.4f}, {ratio.max():.4f}] "
        "leave the MXFP8 quantization bound -- a uniform scale error"
    )


def test_window_ring_wrap_holds_the_latest_records(run: Run) -> None:
    """The recycled block's wrapped slots 0..31 hold the third forward's
    records -- value and scale bytes, in direction and magnitude, within the
    store's MXFP8 precision -- while the same block's slots 32..127 keep
    the first forward's records, and no other layer of the shared window
    leaf was written."""
    swa = run.caches["swa"]
    for phys in range(swa.shape[0]):
        for layer in range(swa.shape[2]):
            if layer == LAYER:
                continue
            assert np.array_equal(
                _records(swa, phys, layer),
                _records(run.initial["swa"], phys, layer),
            ), f"window block {phys} layer {layer} was written through "
            f"layer {LAYER}'s slots"
    table = rope_table()
    page0 = dequant_page(_records(swa, 0, LAYER), W_DATA)
    assert not np.array_equal(
        _records(swa, 0, LAYER), _records(run.initial["swa"], 0, LAYER)
    ), "no window record was stored"
    # The recycled slots: the third forward's tokens 256..287, in magnitude
    # as well as direction.
    latest = page0[:N_TAIL]
    want = rope(
        run.inputs["w_latent"][N_HEAD + N_MID :],
        np.arange(N_HEAD + N_MID, T),
        table,
    )
    _check_records(
        latest, want, "recycled slots hold stale or mis-scaled records"
    )
    # The un-recycled slots of the same block: the first forward's tokens
    # 32..127, which the recycled write must not touch.
    kept = page0[VALID_FROM:N_HEAD]
    want = rope(
        run.inputs["w_latent"][VALID_FROM:N_HEAD],
        np.arange(VALID_FROM, N_HEAD),
        table,
    )
    _check_records(
        kept, want, "the recycled write touched slots it did not own"
    )


def test_skipped_stores_leave_their_slots_untouched(run: Run) -> None:
    """A ``-1`` slot skips its store: the skipped groups' value and scale
    planes, and every plane byte no store named, keep their initial bytes."""
    c2 = run.caches["c2a"]
    # Page 0 (groups 0..63) is block 0: groups 10..19 were skipped; page 2
    # (groups 128..143) is block 2, whose slots past group 143's were
    # never named.
    for phys, lo, hi in ((0, SKIP_LO, SKIP_HI), (2, N_GROUPS - 128, 64)):
        values, scales = _planes(_records(c2, phys, 0), C_DATA)
        iv, isc = _planes(_records(run.initial["c2a"], phys, 0), C_DATA)
        assert np.array_equal(values[lo:hi], iv[lo:hi]), (phys, "values")
        assert np.array_equal(scales[lo:hi], isc[lo:hi]), (phys, "scales")
    # Block 2 of c1a holds positions 256..287 in slots 0..31 only.
    values, scales = _planes(_records(run.caches["c1a"], 2, 0), C_DATA)
    iv, isc = _planes(_records(run.initial["c1a"], 2, 0), C_DATA)
    assert np.array_equal(values[32:], iv[32:]), "unwritten c1a values"
    assert np.array_equal(scales[32:], isc[32:]), "unwritten c1a scales"


def _keys(run: Run) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """The dequantized window pages and c2a/c1a records the lists address.

    The window's page 2 is served from recycled block 0, so entry ``e``
    reads page ``e // 128``'s records, with page 2's living in block 0's
    slots.
    """
    swa0 = dequant_page(_records(run.caches["swa"], 0, LAYER), W_DATA)
    swa1 = dequant_page(_records(run.caches["swa"], 1, LAYER), W_DATA)
    swa = [swa0, swa1, swa0]
    # c2a: pages 0..2 are blocks 0..2; c1a: the same, one record per token.
    c2 = np.concatenate(
        [
            dequant_page(_records(run.caches["c2a"], p, 0), C_DATA)
            for p in range(3)
        ],
        0,
    )[:N_GROUPS]
    c1 = np.concatenate(
        [
            dequant_page(_records(run.caches["c1a"], p, 0), C_DATA)
            for p in range(3)
        ],
        0,
    )[:T]
    return swa, c2, c1


def _oracle(
    run: Run,
    q: np.ndarray,
    positions: Sequence[int],
    sel: np.ndarray,
    comp: np.ndarray,
    ratio: int,
) -> np.ndarray:
    """Float64 attention over the dequantized records, sink included."""
    swa, _, _ = _keys(run)
    sink = run.inputs["sink"].astype(np.float64)
    out = np.zeros((len(positions), NH, HEAD_DIM))
    for t, pos in enumerate(positions):
        # Entry e lives at page e // 128 (page 2 in the recycled block 0),
        # slot e % 128, which the staged stores filled with token e.
        entries = np.arange(int(pos), int(pos) - 128, -1)
        keys = [swa[e // 128][e % 128] for e in entries if e >= VALID_FROM]
        keys += [comp[int(c) // ratio] for c in sel[t] if c >= 0]
        if not keys:
            continue
        k = np.stack(keys)
        s = q[t].astype(np.float64) @ k.T * SCALE
        m = np.maximum(s.max(1), sink)
        p = np.exp(s - m[:, None])
        out[t] = (p @ k) / (p.sum(1) + np.exp(sink - m))[:, None]
    return out


def _check_close(got: np.ndarray, want: np.ndarray, what: str) -> None:
    got = got.astype(np.float64)
    err = np.abs(got - want).max(axis=-1)
    tol = 2e-2 * np.abs(want).max(axis=-1) + 1e-3
    bad = int(np.count_nonzero(err > tol))
    cos = float(
        (got * want).sum() / np.sqrt((got * got).sum() * (want * want).sum())
    )
    assert bad == 0 and cos > 0.9999, (
        f"{what}: {bad} heads past tolerance, cosine {cos:.6f}"
    )


def test_decode_matches_the_oracle(run: Run) -> None:
    """The decode over the recycled ring, the scattered compressed blocks,
    a nonzero SWA floor and one empty selection matches the oracle."""
    _, c2, c1 = _keys(run)
    positions = DECODE_POS
    _check_close(
        run.dec2,
        _oracle(
            run,
            run.inputs["q_decode"],
            positions,
            run.inputs["sel_dec2"],
            c2,
            2,
        ),
        "ratio-2 decode",
    )
    _check_close(
        run.dec1,
        _oracle(
            run,
            run.inputs["q_decode"],
            positions,
            run.inputs["sel_dec1"],
            c1,
            1,
        ),
        "ratio-1 decode",
    )


def test_bounded_prefill_matches_the_oracle(run: Run) -> None:
    """The 128-row prefill -- the first forward's queries, none of whose
    keys the later recycling touched -- matches the oracle row by row,
    including the rows whose whole window lies below the floor and the
    first row whose every list entry is ``-1``. The wrap state itself is
    checked directly by :func:`test_window_ring_wrap_holds_the_latest_records`."""
    _, c2, _ = _keys(run)
    _check_close(
        run.pre2,
        _oracle(
            run,
            run.inputs["q_prefill"],
            list(range(N_PRE)),
            run.inputs["sel_pre2"],
            c2,
            2,
        ),
        "ratio-2 prefill",
    )
