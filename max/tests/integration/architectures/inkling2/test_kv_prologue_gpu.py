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
"""The fused K/V prologue matches MAX's composed ops on a real paged cache.

The composed path is the one ``Inkling2Attention`` runs without the flag:
``short_conv_ring`` on K and on V, ``l2_clip`` on K's heads,
``unweighted_rms_norm`` on V's heads, then ``store_k_cache_ragged`` and
``store_v_cache_ragged``. One compiled graph runs both paths, each on its own
copy of the cache and the rings, and every block of the cache and every ring
entry is compared afterwards, so a write to the wrong page, layer, slot or
ring entry fails as surely as a wrong value. The row and batch counts are
symbolic, so every case of one ring length shares a compile.
"""

from __future__ import annotations

import dataclasses
import functools
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from max import functional as F
from max import tree
from max.driver import CPU, Accelerator, Buffer
from max.driver.bfloat16 import (
    bfloat16_as_uint16_to_float32,
    float32_to_bfloat16_as_uint16,
)
from max.dtype import DType
from max.graph import BufferType, DeviceRef, TensorType
from max.nn import Module
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.kernels import store_k_cache_ragged, store_v_cache_ragged
from max.nn.kv_cache import KVCacheInputsPerDevice, MHAKVCacheParams
from max.pipelines.architectures.inkling2.layers.attention import (
    _CLIP_EPS,
    l2_clip,
)
from max.pipelines.architectures.inkling2.layers.kv_prologue import kv_prologue
from max.pipelines.architectures.inkling2.layers.norm import (
    unweighted_rms_norm,
)
from max.pipelines.architectures.inkling2.layers.short_convolution import (
    ShortConvolution,
    SiteState,
    _convolve,
)
from max.tensor import Tensor
from test_common.simple_kv_cache import paged_kv_cache_inputs

# Inkling 2 Small's local layers at TP4: two 128-wide KV heads per rank, a
# four-tap conv, the checkpoint's clip threshold and norm epsilon.
HEADS = 2
HEAD_DIM = 128
KV_DIM = HEADS * HEAD_DIM
KERNEL = 4
CLIP = 22.627416997969522
EPS = 1e-6
PAGE_SIZE = 16
CACHE_LAYERS = 3
CACHE_LAYER = 1
SLOT_LAYERS = 2
K_SLOT_ROW = 1
V_SLOT_ROW = 0
MAX_BATCH = 16
"""The slot tables' width: wider than any case's batch, so a table indexed
with the launch's row count instead of its own stride reads the wrong slot."""
NUM_SLOTS = SLOT_LAYERS * MAX_BATCH
TOTAL_PAGES = 1280
"""Enough pages for an 8192-token prefill beside other sequences."""

# Both sides scale the same bfloat16 conv output by a factor from the same
# float32 head sum, associated the same way, so every value is bit-identical.
MAX_ULP = 0

_SENTINEL = 0x7F7F
"""bfloat16 bits of a large finite value that neither path can produce."""


def _kv_params() -> MHAKVCacheParams:
    return MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=HEADS,
        head_dim=HEAD_DIM,
        num_layers=CACHE_LAYERS,
        page_size=PAGE_SIZE,
        devices=[DeviceRef.GPU()],
    )


def _bf16(x: np.ndarray) -> np.ndarray:
    """Rounds to the nearest bfloat16, kept as float32."""
    bits = float32_to_bfloat16_as_uint16(x.astype(np.float32))
    return bfloat16_as_uint16_to_float32(bits)


def _ulps(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance in bfloat16 ulps between two arrays of bfloat16 values."""

    def ordered(x: np.ndarray) -> np.ndarray:
        bits = float32_to_bfloat16_as_uint16(x).astype(np.int64)
        return np.where(bits & 0x8000, -(bits & 0x7FFF), bits)

    return np.abs(ordered(a) - ordered(b))


class _BothPaths(Module[..., Tensor]):
    """The composed path into one cache and rings, the fused op into another."""

    def __init__(self) -> None:
        self.kv_params = _kv_params()

    def forward(self, *inputs: Any) -> Tensor:
        k_in, v_in, k_taps, v_taps, offsets, positions, k_rows, v_rows = inputs[
            :8
        ]
        rings = inputs[8:12]
        leaves = inputs[12:]
        half = len(leaves) // 2
        want_kv, got_kv = (
            self._cache(leaves[:half]),
            self._cache(leaves[half:]),
        )
        layer = F.constant(CACHE_LAYER, DType.uint32, device=CPU())

        def site(ring: Tensor, rows: Tensor, row: int) -> SiteState:
            return SiteState(
                ring=ring,
                rows=rows,
                layer_row=F.constant(row, DType.uint32, device=CPU()),
            )

        def conv(taps: Tensor) -> ShortConvolution:
            return cast(
                ShortConvolution,
                SimpleNamespace(
                    weight=taps.reshape([KV_DIM, 1, KERNEL]),
                    kernel_size=KERNEL,
                ),
            )

        k = _convolve(
            conv(k_taps),
            k_in,
            site(rings[0], k_rows, K_SLOT_ROW),
            offsets,
            positions,
        )
        v = _convolve(
            conv(v_taps),
            v_in,
            site(rings[1], v_rows, V_SLOT_ROW),
            offsets,
            positions,
        )
        k = l2_clip(k.reshape([-1, HEADS, HEAD_DIM]), CLIP)
        v = unweighted_rms_norm(
            v.cast(DType.float32).reshape([-1, HEADS, HEAD_DIM]), EPS
        ).cast(v.dtype)
        store_k_cache_ragged(want_kv, k, offsets, layer)
        store_v_cache_ragged(want_kv, v, offsets, layer)

        kv_prologue(
            k_in,
            v_in,
            k_taps,
            v_taps,
            site(rings[2], k_rows, K_SLOT_ROW),
            site(rings[3], v_rows, V_SLOT_ROW),
            got_kv,
            positions,
            input_row_offsets=offsets,
            layer_idx=layer,
            clip=CLIP,
            clip_eps=_CLIP_EPS,
            eps=EPS,
        )
        return k_in

    def _cache(self, leaves: tuple[Any, ...]) -> PagedCacheValues:
        (kv,) = self.kv_params.unflatten_kv_inputs(iter(leaves))
        return PagedCacheValues(
            **{f.name: getattr(kv, f.name) for f in dataclasses.fields(kv)}
        )


@functools.cache
def _compiled(ring_len: int) -> Any:
    gpu = DeviceRef.GPU()
    module = _BothPaths()
    kv_types = module.kv_params.flattened_kv_inputs()
    ring = BufferType(DType.float32, [NUM_SLOTS, ring_len, KV_DIM], gpu)
    table = TensorType(DType.uint32, [SLOT_LAYERS, MAX_BATCH], device=gpu)
    act = TensorType(DType.bfloat16, ["rows", KV_DIM], device=gpu)
    taps = TensorType(DType.bfloat16, [KV_DIM, KERNEL], device=gpu)
    return module.compile(
        act,
        act,
        taps,
        taps,
        TensorType(DType.uint32, ["offsets"], device=gpu),
        TensorType(DType.uint32, ["rows"], device=gpu),
        table,
        table,
        ring,
        ring,
        ring,
        ring,
        *kv_types,
        *kv_types,
    )


def _to_bf16_buffer(x: np.ndarray, device: Accelerator) -> Buffer:
    bits = float32_to_bfloat16_as_uint16(x.astype(np.float32))
    return Buffer.from_numpy(bits).view(DType.bfloat16).to(device)


def _read_bf16(buf: Buffer) -> np.ndarray:
    return bfloat16_as_uint16_to_float32(
        buf.to(CPU()).view(DType.uint16).to_numpy()
    )


def _run(
    prompt_lens: list[int], cache_lengths: list[int], ring_len: int
) -> None:
    rng = np.random.default_rng(
        sum(prompt_lens) * 1000 + sum(cache_lengths) + ring_len
    )
    device = Accelerator(0)
    rows = sum(prompt_lens)
    kv_params = _kv_params()

    def cache() -> KVCacheInputsPerDevice[Buffer, Buffer]:
        return paged_kv_cache_inputs(
            kv_params,
            prompt_lens,
            cache_lengths=cache_lengths,
            total_num_pages=TOTAL_PAGES,
        )

    # Pages handed out out of order, so a store that ignores the lookup table
    # lands on the wrong page.
    want_kv, got_kv = cache(), cache()
    lut = want_kv.lookup_table.to(CPU()).to_numpy()
    perm = rng.permutation(TOTAL_PAGES).astype(np.uint32)
    lut = np.where(
        lut < TOTAL_PAGES, perm[np.minimum(lut, TOTAL_PAGES - 1)], lut
    ).astype(np.uint32)
    # Every cache element starts as a sentinel no store writes, so a write
    # anywhere, including a legitimate zero, is visible.
    blocks_shape = want_kv.kv_blocks.shape
    want_kv, got_kv = (
        dataclasses.replace(
            kv,
            lookup_table=Buffer.from_numpy(lut).to(device),
            kv_blocks=Buffer.from_numpy(
                np.full(blocks_shape, _SENTINEL, np.uint16)
            )
            .view(DType.bfloat16)
            .to(device),
        )
        for kv in (want_kv, got_kv)
    )

    slots = [
        rng.permutation(NUM_SLOTS)
        .astype(np.uint32)
        .reshape(SLOT_LAYERS, MAX_BATCH)
        for _ in range(2)
    ]
    rings = [
        _bf16(rng.standard_normal((NUM_SLOTS, ring_len, KV_DIM)))
        for _ in range(2)
    ]
    ring_bufs = [
        Buffer.from_numpy(r.copy()).to(device) for r in (*rings, *rings)
    ]

    # Alternate small and large tokens, so some K heads clip and some do not.
    scale = np.where(np.arange(rows) % 2 == 0, 0.5, 5.0)[:, None]
    k_np = rng.standard_normal((rows, KV_DIM)) * scale
    v_np = rng.standard_normal((rows, KV_DIM)) * scale
    taps = [0.5 * rng.standard_normal((KV_DIM, KERNEL)) for _ in range(2)]
    offsets = np.concatenate([[0], np.cumsum(prompt_lens)]).astype(np.uint32)
    positions = np.concatenate(
        [
            np.arange(c, c + n, dtype=np.uint32)
            for c, n in zip(cache_lengths, prompt_lens, strict=True)
        ]
    )

    buffers = [
        _to_bf16_buffer(k_np, device),
        _to_bf16_buffer(v_np, device),
        _to_bf16_buffer(taps[0], device),
        _to_bf16_buffer(taps[1], device),
        Buffer.from_numpy(offsets).to(device),
        Buffer.from_numpy(positions).to(device),
        Buffer.from_numpy(slots[0]).to(device),
        Buffer.from_numpy(slots[1]).to(device),
        *ring_bufs,
        *tree.leaves(want_kv),
        *tree.leaves(got_kv),
    ]
    _compiled(ring_len)(*(Tensor(storage=b) for b in buffers))

    want_blocks = _read_bf16(want_kv.kv_blocks)
    got_blocks = _read_bf16(got_kv.kv_blocks)
    case = f"prompt_lens={prompt_lens} cache_lengths={cache_lengths}"
    sentinel = bfloat16_as_uint16_to_float32(np.array([_SENTINEL], np.uint16))
    written = want_blocks != sentinel
    others = [i for i in range(CACHE_LAYERS) if i != CACHE_LAYER]
    counts = [
        int((b[:, :, CACHE_LAYER] != sentinel).sum())
        for b in (want_blocks, got_blocks)
    ]
    assert counts == [2 * rows * KV_DIM] * 2, f"{case}: {counts}"
    assert not written[:, :, others].any(), case
    for plane, name in ((0, "K"), (1, "V")):
        ulps = _ulps(got_blocks[:, plane], want_blocks[:, plane])
        mask = written[:, plane]
        print(
            f"{case} {name}: max {ulps.max()} ulp,"
            f" {int((ulps[mask] == 0).sum())}/{int(mask.sum())} written"
            " values bit-identical"
        )
        assert ulps.max() <= MAX_ULP, f"{case} {name}: {ulps.max()} ulp"
    for i, name in enumerate("KV"):
        want = ring_bufs[i].to(CPU()).to_numpy()
        got = ring_bufs[2 + i].to(CPU()).to_numpy()
        assert not np.array_equal(want, rings[i]), f"{case} {name} ring"
        np.testing.assert_array_equal(got, want, err_msg=f"{case} {name} ring")


@pytest.mark.parametrize(
    ("prompt_lens", "cache_lengths"),
    [
        # One token, mid-page.
        ([1], [37]),
        # Eight tokens: positions below `kernel - 1` read zeros for the
        # missing history, and 15/16 sit either side of a page boundary.
        ([1] * 8, [0, 1, 2, 3, 15, 16, 47, 130]),
    ],
)
def test_kv_prologue_decode(
    prompt_lens: list[int], cache_lengths: list[int]
) -> None:
    _run(prompt_lens, cache_lengths, KERNEL - 1)


def test_kv_prologue_ragged_chunks() -> None:
    """Prefill chunks beside decode rows, with a verify-sized ring.

    A fresh prefill, a decode row at a sub-kernel position, a chunk across a
    page boundary, a chunk longer than the ring (only its tail commits), a
    chunk shorter than the kernel's history, and a first token.
    """
    _run([5, 1, 3, 9, 2, 1], [0, 2, 14, 1, 30, 0], KERNEL - 1 + 3)


def test_kv_prologue_ragged_chunks_short_ring() -> None:
    """The same mix with the ring at the kernel's history and no more."""
    _run([5, 1, 3, 9, 2, 1], [0, 2, 14, 1, 30, 0], KERNEL - 1)


def test_kv_prologue_long_prefill() -> None:
    """A whole 8192-token prefill, then the decode step that reads its ring.

    Two million values per plane, so a head sum associated differently from
    the composed ops' shows up as a rounding flip here even when the short
    cases happen to match.
    """
    _run([8192, 1], [0, 8192], KERNEL - 1)


def test_kv_prologue_long_prefill_chunked() -> None:
    """A second prefill chunk, a decode row and a fresh prefill together."""
    _run([4096, 1, 3000], [4096, 8191, 0], KERNEL - 1)
