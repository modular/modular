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
"""Bit-exactness test for ``kv_cache_gather_rows_ragged`` against numpy.

The reference resolves every slot through the lookup table by hand, so the
kernel's paging, row-to-request mapping and key/value selection are checked
independently of the Mojo cache types. The op is a pure copy, so CPU and GPU
must both match the block buffer bit for bit.
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
from max.nn.kernels import (
    KEY_CACHE_INDEX,
    VALUE_CACHE_INDEX,
    kv_cache_gather_rows_ragged,
)
from max.nn.kv_cache import KVCacheParams, MHAKVCacheParams, MLAKVCacheParams
from test_common.simple_kv_cache import paged_kv_cache_inputs

PAGE_SIZE = 128
NUM_LAYERS = 3
LAYER = 1
TOTAL_PAGES = 16


@dataclasses.dataclass(frozen=True)
class Leaf:
    """A leaf layout the DeepSeek-V4 cache reads rows out of."""

    name: str
    dtype: DType
    head_dim: int
    mla: bool
    slots_per_page: int | None
    window_size: int | None

    def params(self, device: DeviceRef) -> KVCacheParams:
        if self.mla:
            return MLAKVCacheParams(
                dtype=self.dtype,
                head_dim=self.head_dim,
                num_layers=NUM_LAYERS,
                devices=[device],
                page_size=PAGE_SIZE,
                slots_per_page=self.slots_per_page,
                window_size=self.window_size,
                num_q_heads=8,
            )
        return MHAKVCacheParams(
            dtype=self.dtype,
            head_dim=self.head_dim,
            num_layers=NUM_LAYERS,
            devices=[device],
            page_size=PAGE_SIZE,
            slots_per_page=self.slots_per_page,
            window_size=self.window_size,
            n_kv_heads=1,
        )


LEAVES = [
    # Compressed zone: 32 entries per 128-token page, addressed by entry.
    Leaf("zone_bf16", DType.bfloat16, 128, True, PAGE_SIZE // 4, None),
    # Sliding-window latent, addressed by absolute token position.
    Leaf("window_bf16", DType.bfloat16, 512, True, None, 128),
    # Compressor open state: float32 K and V, addressed by position.
    Leaf("state_f32", DType.float32, 256, False, None, 8),
]


def _device(name: str) -> tuple[Device, DeviceRef]:
    if name == "cpu":
        return CPU(), DeviceRef.CPU()
    return Accelerator(), DeviceRef.GPU()


def _random_blocks(shape: tuple[int, ...], dtype: DType) -> np.ndarray:
    rng = np.random.default_rng(0)
    values = rng.standard_normal(shape).astype(np.float32)
    if dtype == DType.bfloat16:
        return values.astype(ml_dtypes.bfloat16).view(np.uint16)
    return values


def _to_buffer(bits: np.ndarray, dtype: DType, device: Device) -> Buffer:
    buffer = Buffer.from_numpy(bits)
    if dtype == DType.bfloat16:
        buffer = buffer.view(DType.bfloat16)
    return buffer.to(device)


def _to_bits(buffer: Buffer, dtype: DType) -> np.ndarray:
    host = buffer.to(CPU())
    if dtype == DType.bfloat16:
        host = host.view(DType.uint16)
    return host.to_numpy()


# (rows per request, tokens per request, slots per row)
BATCHES = [
    # Ragged prefill rows, two requests.
    ([5, 3], [200, 40], 48),
    # One row per request, as in decode.
    ([1, 1, 1], [300, 5, 130], 16),
    # A request with no rows between two with rows.
    ([4, 0, 2], [64, 10, 257], 7),
]


@pytest.mark.parametrize("device_name", ["cpu", "gpu"])
@pytest.mark.parametrize("leaf", LEAVES, ids=lambda leaf: leaf.name)
def test_gather_rows_matches_numpy(device_name: str, leaf: Leaf) -> None:
    device, device_ref = _device(device_name)
    params = leaf.params(device_ref)
    slots_per_page = params.slots_per_page
    assert slots_per_page is not None
    ratio = PAGE_SIZE // slots_per_page
    kv_sides = [KEY_CACHE_INDEX]
    if not leaf.mla:
        kv_sides.append(VALUE_CACHE_INDEX)

    # Symbolic shapes, so one compiled graph serves every batch below.
    with Graph(
        "kv_cache_gather_rows",
        input_types=[
            TensorType(DType.int32, ["rows", "num_slots"], device_ref),
            TensorType(DType.uint32, ["batch_plus_one"], device_ref),
            *params.flattened_kv_inputs(),
        ],
    ) as graph:
        slots_in, offsets_in, *rest = graph.inputs
        collection = params.unflatten_kv_inputs(iter(rest))[0]
        layer = ops.constant(LAYER, DType.uint32, DeviceRef.CPU())
        graph.output(
            *(
                kv_cache_gather_rows_ragged(
                    collection,
                    slots_in.tensor,
                    offsets_in.tensor,
                    layer,
                    key_or_value=kv,
                )
                for kv in kv_sides
            )
        )
    model = InferenceSession(devices=[device]).load(graph)

    rng = np.random.default_rng(1)
    for row_counts, seq_lens, num_slots in BATCHES:
        # A zone leaf is written at ``cache_lengths // ratio``; its live slots
        # are the entries of the request's tokens.
        live = [max(1, n // ratio) for n in seq_lens]
        inputs = paged_kv_cache_inputs(
            params, seq_lens, total_num_pages=TOTAL_PAGES
        )
        blocks = _random_blocks(tuple(inputs.kv_blocks.shape), leaf.dtype)
        inputs = dataclasses.replace(
            inputs, kv_blocks=_to_buffer(blocks, leaf.dtype, device)
        )
        lut = inputs.lookup_table.to(CPU()).to_numpy()

        row_offsets = np.zeros(len(row_counts) + 1, dtype=np.uint32)
        row_offsets[1:] = np.cumsum(row_counts)
        rows = int(row_offsets[-1])
        batch_of_row = (
            np.searchsorted(row_offsets, np.arange(rows), "right") - 1
        )
        slots = np.zeros((rows, num_slots), dtype=np.int32)
        for r in range(rows):
            slots[r] = rng.integers(0, live[batch_of_row[r]], size=num_slots)

        results = model(
            Buffer.from_numpy(slots).to(device),
            Buffer.from_numpy(row_offsets).to(device),
            *tree.leaves(inputs),
        )

        for kv, result in zip(kv_sides, results, strict=True):
            want = np.zeros(
                (rows, num_slots, leaf.head_dim), dtype=blocks.dtype
            )
            for r in range(rows):
                for j, slot in enumerate(slots[r]):
                    page = int(lut[batch_of_row[r], slot // slots_per_page])
                    want[r, j] = blocks[
                        page, kv, LAYER, slot % slots_per_page, 0
                    ]
            np.testing.assert_array_equal(_to_bits(result, leaf.dtype), want)
