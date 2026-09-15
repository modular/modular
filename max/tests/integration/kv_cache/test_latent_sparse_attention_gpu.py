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
"""Numerical test for ``latent_sparse_attention_ragged`` against numpy.

Two paged leaves of a single shared latent head: a sliding-window leaf paged
by token position and a compressed leaf paged by entry (``slots_per_page``).
The reference resolves every key through the lookup tables by hand, so the
kernel's paging, ragged batching, ``-1`` skipping and sink handling are all
checked independently of the Mojo cache types.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.kernels import latent_sparse_attention_ragged
from max.nn.kv_cache import MLAKVCacheParams, flatten_kv_inputs_per_device
from test_common.simple_kv_cache import paged_kv_cache_inputs

HEAD_DIM = 512
NUM_HEADS = 64
WINDOW = 128
PAGE_SIZE = 128
RATIO = 4
NUM_LAYERS = 2
TOTAL_PAGES = 24


def _leaf_params(
    *, slots_per_page: int | None, window: int | None
) -> MLAKVCacheParams:
    return MLAKVCacheParams(
        dtype=DType.float32,
        head_dim=HEAD_DIM,
        num_layers=NUM_LAYERS,
        devices=[DeviceRef.GPU()],
        page_size=PAGE_SIZE,
        num_q_heads=NUM_HEADS,
        slots_per_page=slots_per_page,
        window_size=window,
    )


def _resolve(lut: np.ndarray, b: int, idx: int, slots: int) -> tuple[int, int]:
    """Physical (block, slot) of logical slot ``idx`` in sequence ``b``."""
    return int(lut[b, idx // slots]), idx % slots


def _reference(
    q: np.ndarray,
    row_offsets: np.ndarray,
    cache_lengths: np.ndarray,
    comp_idx: np.ndarray,
    sink: np.ndarray,
    swa_blocks: np.ndarray,
    swa_lut: np.ndarray,
    comp_blocks: np.ndarray,
    comp_lut: np.ndarray,
    layer_swa: int,
    layer_comp: int,
    scale: float,
) -> np.ndarray:
    total = q.shape[0]
    out = np.zeros_like(q, dtype=np.float64)
    comp_slots = comp_blocks.shape[3]
    for t in range(total):
        b = int(np.searchsorted(row_offsets, t, side="right") - 1)
        pos = int(cache_lengths[b]) + t - int(row_offsets[b])
        keys = []
        for p in range(max(0, pos - WINDOW + 1), pos + 1):
            blk, slot = _resolve(swa_lut, b, p, PAGE_SIZE)
            keys.append(swa_blocks[blk, 0, layer_swa, slot, 0])
        for e in comp_idx[t]:
            if e < 0:
                continue
            blk, slot = _resolve(comp_lut, b, int(e), comp_slots)
            keys.append(comp_blocks[blk, 0, layer_comp, slot, 0])
        k = np.stack(keys).astype(np.float64)  # [n, d]
        s = (q[t].astype(np.float64) @ k.T) * scale  # [heads, n]
        m = s.max(axis=-1, keepdims=True)
        p = np.exp(s - m)
        den = p.sum(axis=-1, keepdims=True) + np.exp(
            sink[:, None].astype(np.float64) - m
        )
        out[t] = (p @ k) / den
    return out


@pytest.mark.parametrize(
    ("prompt_lens", "cache_lengths", "num_comp"),
    [
        # Prefill from empty: rows below the window and rows past it.
        ([200, 40], [0, 0], 16),
        # Decode: one row per sequence, deep into the window / compressed zone.
        ([1, 1, 1], [300, 5, 130], 48),
        # Chunked extend with an empty compressed list.
        ([64, 10], [70, 500], 0),
    ],
)
def test_latent_sparse_attention_matches_numpy(
    prompt_lens: list[int], cache_lengths: list[int], num_comp: int
) -> None:
    device = Accelerator()
    rng = np.random.default_rng(0)
    batch = len(prompt_lens)
    total = sum(prompt_lens)
    scale = HEAD_DIM**-0.5
    layer_swa, layer_comp = 1, 0

    swa_params = _leaf_params(slots_per_page=None, window=WINDOW)
    comp_params = _leaf_params(slots_per_page=PAGE_SIZE // RATIO, window=None)

    swa_inputs = paged_kv_cache_inputs(
        swa_params,
        prompt_lens,
        cache_lengths=cache_lengths,
        total_num_pages=TOTAL_PAGES,
    )
    # The compressed leaf is addressed by entry: its lengths are in entries.
    comp_prompt = [max(1, n // RATIO) for n in prompt_lens]
    comp_cache = [c // RATIO for c in cache_lengths]
    comp_inputs = paged_kv_cache_inputs(
        comp_params,
        comp_prompt,
        cache_lengths=comp_cache,
        total_num_pages=TOTAL_PAGES,
    )

    swa_blocks = rng.standard_normal(swa_inputs.kv_blocks.shape).astype(
        np.float32
    )
    comp_blocks = rng.standard_normal(comp_inputs.kv_blocks.shape).astype(
        np.float32
    )
    swa_inputs = dataclasses.replace(
        swa_inputs, kv_blocks=Buffer.from_numpy(swa_blocks).to(device)
    )
    comp_inputs = dataclasses.replace(
        comp_inputs, kv_blocks=Buffer.from_numpy(comp_blocks).to(device)
    )
    swa_lut = swa_inputs.lookup_table.to_numpy()
    comp_lut = comp_inputs.lookup_table.to_numpy()

    row_offsets = np.zeros(batch + 1, dtype=np.uint32)
    row_offsets[1:] = np.cumsum(prompt_lens)
    q = rng.standard_normal((total, NUM_HEADS, HEAD_DIM)).astype(np.float32)
    sink = rng.standard_normal(NUM_HEADS).astype(np.float32)

    # Each row may only reference entries that exist for its sequence; pad
    # the tail of every list with -1 so the skip path is exercised too.
    comp_idx = np.full((total, num_comp), -1, dtype=np.int32)
    for t in range(total):
        b = int(np.searchsorted(row_offsets, t, side="right") - 1)
        available = comp_cache[b] + comp_prompt[b]
        n = min(num_comp, available) if num_comp else 0
        if n:
            picks = rng.choice(available, size=n, replace=False)
            comp_idx[t, : n - (t % 2)] = picks[: n - (t % 2)]

    with Graph(
        "latent_sparse_attention",
        input_types=[
            TensorType(
                DType.float32, [total, NUM_HEADS, HEAD_DIM], DeviceRef.GPU()
            ),
            TensorType(DType.uint32, [batch + 1], DeviceRef.GPU()),
            TensorType(DType.int32, [total, num_comp], DeviceRef.GPU()),
            TensorType(DType.float32, [NUM_HEADS], DeviceRef.GPU()),
            *swa_params.flattened_kv_inputs(),
            *comp_params.flattened_kv_inputs(),
        ],
    ) as graph:
        q_in, offs_in, idx_in, sink_in, *rest = graph.inputs
        it = iter(rest)
        swa_collection = swa_params.unflatten_kv_inputs(it).inputs[0]
        comp_collection = comp_params.unflatten_kv_inputs(it).inputs[0]
        out = latent_sparse_attention_ragged(
            q_in.tensor,
            offs_in.tensor,
            idx_in.tensor,
            sink_in.tensor,
            swa_collection,
            comp_collection,
            ops.constant(layer_swa, DType.uint32, DeviceRef.CPU()),
            ops.constant(layer_comp, DType.uint32, DeviceRef.CPU()),
            scale=scale,
            window=WINDOW,
        )
        graph.output(out)

    model = InferenceSession(devices=[device]).load(graph)
    (result,) = model(
        Buffer.from_numpy(q).to(device),
        Buffer.from_numpy(row_offsets).to(device),
        Buffer.from_numpy(comp_idx).to(device),
        Buffer.from_numpy(sink).to(device),
        *flatten_kv_inputs_per_device(swa_inputs),
        *flatten_kv_inputs_per_device(comp_inputs),
    )
    got = result.to_numpy()

    want = _reference(
        q,
        row_offsets,
        np.array(cache_lengths),
        comp_idx,
        sink,
        swa_blocks,
        swa_lut,
        comp_blocks,
        comp_lut,
        layer_swa,
        layer_comp,
        scale,
    )
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)
