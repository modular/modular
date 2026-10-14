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
"""``indexer_score_ragged`` against a numpy reference that pages by hand.

The reference resolves every candidate column to its entry, looks the entry
up through the lookup table and scores it in float64, so the kernel's column
numbering, liveness rule, paging and head sum are checked independently of
the Mojo cache types. Values sit on a coarse grid, as the model's FP4 ones
do, which makes every dot product exact in float32; the weighting and the
float32 head sum still round, hence a tolerance rather than bit equality.
"""

from __future__ import annotations

import dataclasses

import ml_dtypes
import numpy as np
import pytest
from max import tree
from max.driver import CPU, Accelerator, Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.kernels import indexer_score_ragged
from max.nn.kv_cache import MLAKVCacheParams
from test_common.simple_kv_cache import paged_kv_cache_inputs

PAGE_SIZE = 128
RATIO = 4
HEAD_DIM = 128
NUM_LAYERS = 3
LAYER = 1
TOTAL_PAGES = 24
# ``ceil(max_seq_len / ratio)`` for a max_seq_len of 1024.
CAP = 256


def _device(name: str) -> tuple[Device, DeviceRef]:
    if name == "cpu":
        return CPU(), DeviceRef.CPU()
    return Accelerator(), DeviceRef.GPU()


def _grid(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    """Multiples of 1/2 in [-6, 6] with power-of-two scales, like FP4."""
    values = rng.integers(-12, 13, size=shape).astype(np.float32) / 2
    scales = np.exp2(rng.integers(-2, 3, size=shape[:-1] + (1,)))
    return (values * scales).astype(np.float32)


# (tokens already cached, tokens in this chunk) per request.
BATCHES = [
    # Prefill of two requests; the first spans pages.
    [(0, 300), (0, 37)],
    # Decode rows with long and short histories.
    [(900, 1), (3, 1), (517, 1)],
    # Chunked prefill resumes, one with nothing closed yet.
    [(128, 64), (2, 9)],
]


@pytest.mark.parametrize("device_name", ["cpu", "gpu"])
@pytest.mark.parametrize("q_dtype", [DType.bfloat16, DType.float32])
@pytest.mark.parametrize("num_heads", [32, 64])
def test_indexer_score_matches_numpy(
    device_name: str, q_dtype: DType, num_heads: int
) -> None:
    device, device_ref = _device(device_name)
    params = MLAKVCacheParams(
        dtype=DType.bfloat16,
        head_dim=HEAD_DIM,
        num_layers=NUM_LAYERS,
        devices=[device_ref],
        page_size=PAGE_SIZE,
        slots_per_page=PAGE_SIZE // RATIO,
        num_q_heads=8,
    )
    spp = PAGE_SIZE // RATIO

    with Graph(
        "indexer_score",
        input_types=[
            TensorType(q_dtype, ["rows", num_heads, HEAD_DIM], device_ref),
            TensorType(DType.float32, ["rows", num_heads], device_ref),
            TensorType(DType.uint32, ["batch_plus_one"], device_ref),
            TensorType(DType.int32, ["batch"], device_ref),
            TensorType(DType.int32, ["rows"], device_ref),
            *params.flattened_kv_inputs(),
        ],
    ) as graph:
        q_in, w_in, offsets_in, base_in, cutoff_in, *rest = graph.inputs
        collection = params.unflatten_kv_inputs(iter(rest))[0]
        graph.output(
            indexer_score_ragged(
                q_in.tensor,
                w_in.tensor,
                offsets_in.tensor,
                base_in.tensor,
                cutoff_in.tensor,
                collection,
                ops.constant(LAYER, DType.uint32, DeviceRef.CPU()),
                num_candidates=2 * CAP,
            )
        )
    model = InferenceSession(devices=[device]).load(graph)

    rng = np.random.default_rng(0)
    for batch in BATCHES:
        cached = [p for p, _ in batch]
        chunk = [s for _, s in batch]
        inputs = paged_kv_cache_inputs(
            params,
            chunk,
            cache_lengths=cached,
            total_num_pages=TOTAL_PAGES,
        )
        blocks = _grid(rng, tuple(inputs.kv_blocks.shape)).astype(
            ml_dtypes.bfloat16
        )
        buffer = Buffer.from_numpy(blocks.view(np.uint16)).view(DType.bfloat16)
        inputs = dataclasses.replace(inputs, kv_blocks=buffer.to(device))
        lut = inputs.lookup_table.to(CPU()).to_numpy()

        offsets = np.zeros(len(batch) + 1, dtype=np.uint32)
        offsets[1:] = np.cumsum(chunk)
        rows = int(offsets[-1])
        bid = np.searchsorted(offsets, np.arange(rows), "right") - 1
        pos = np.array(cached)[bid] + np.arange(rows) - offsets[bid]
        base = (np.array(cached) // RATIO).astype(np.int32)
        cutoff = ((pos + 1) // RATIO).astype(np.int32)

        q = _grid(rng, (rows, num_heads, HEAD_DIM))
        weights = rng.standard_normal((rows, num_heads)).astype(np.float32)
        if q_dtype == DType.bfloat16:
            q_buf = Buffer.from_numpy(
                q.astype(ml_dtypes.bfloat16).view(np.uint16)
            ).view(DType.bfloat16)
        else:
            q_buf = Buffer.from_numpy(q)

        (result,) = model(
            q_buf.to(device),
            Buffer.from_numpy(weights).to(device),
            Buffer.from_numpy(offsets).to(device),
            Buffer.from_numpy(base).to(device),
            Buffer.from_numpy(cutoff).to(device),
            *tree.leaves(inputs),
        )
        got = result.to(CPU()).to_numpy()

        want = np.zeros((rows, 2 * CAP), dtype=np.float64)
        for t in range(rows):
            b = bid[t]
            for c in range(2 * CAP):
                e = c if c < CAP else base[b] + c - CAP
                live = c < base[b] if c < CAP else e < cutoff[t]
                if not live:
                    continue
                page = int(lut[b, e // spp])
                k = blocks[page, 0, LAYER, e % spp, 0].astype(np.float64)
                dots = q[t].astype(np.float64) @ k
                want[t, c] = np.sum(np.maximum(dots, 0) * weights[t])

        np.testing.assert_array_equal(got[want == 0], 0)
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-3)
