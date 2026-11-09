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
"""Mixed-step split of ragged MHA (B200).

A mixed step batches decode rows (``1 + K`` tokens each) with prefill rows.
With a split record on the KV leaf, ``attention.mha.ragged.paged`` cuts the batch
into a head and a tail view and runs each on its own route, into one output.
The oracle: the split output equals each view's rows run alone, bit for bit.
A zeros record or a malformed one keeps the single launch, so it equals the
call without a record (which sends zeros).

The shape is one attention layer on one GPU, 24 query heads over 4 KV heads:
depth 256 with FP8 or BF16 query and KV, and with BF16 at depth 512 and
under a sliding-window mask.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import pytest
import torch
from max._kv_cache_ops import mha_decode_num_partitions
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    Graph,
    TensorType,
    ops,
)
from max.nn._graph_ops import flash_attention_ragged
from max.nn.attention_mask import MHAMaskVariant
from max.nn.kv_cache import (
    MHAKVCacheParams,
    PagedCacheValues,
    packed_page_stride,
    padded_lut_cols,
)
from test_common.graph_utils import is_b100_b200

NUM_Q_HEADS = 24
NUM_KV_HEADS = 4
PAGE_SIZE = 128
NUM_DRAFT_TOKENS = 3
SPLIT_SIZE = 7
NUM_PAGES = 320
TORCH_DTYPES = {
    DType.float8_e4m3fn: torch.float8_e4m3fn,
    DType.bfloat16: torch.bfloat16,
}


@dataclass(frozen=True)
class Shape:
    kv_dtype: DType
    head_dim: int = 256
    local_window_size: int = -1


@dataclass(frozen=True)
class Row:
    """One request: cached tokens before it, new tokens in it."""

    prefix: int
    width: int

    @property
    def valid_length(self) -> int:
        return self.prefix + self.width


def _record(rows: Sequence[Row], head_rows: int) -> np.ndarray:
    """The record the host rule fills for a cut after ``head_rows``."""
    head, tail = rows[:head_rows], rows[head_rows:]
    head_is_decode = max(r.width for r in head) <= 8
    prefill = tail if head_is_decode else head
    return np.array(
        [
            head_rows,
            sum(r.width for r in head),
            max(r.width for r in head),
            max(r.valid_length for r in head),
            max(r.width for r in tail),
            max(r.valid_length for r in tail),
            sum(r.valid_length for r in prefill),
        ],
        dtype=np.uint32,
    )


def _build(session: InferenceSession, with_record: bool, shape: Shape) -> Model:
    gpu, cpu = DeviceRef.GPU(), DeviceRef.CPU()
    kv_dtype, head_dim = shape.kv_dtype, shape.head_dim
    input_types: list[BufferType | TensorType] = [
        TensorType(kv_dtype, ["total_q", NUM_Q_HEADS, head_dim], device=gpu),
        TensorType(DType.uint32, ["batch_p1"], device=gpu),
        BufferType(
            kv_dtype,
            ["num_pages", 2, 1, PAGE_SIZE, NUM_KV_HEADS, head_dim],
            device=gpu,
        ),
        TensorType(DType.uint32, ["batch"], device=gpu),
        TensorType(DType.uint32, ["batch", "lut_cols"], device=gpu),
        TensorType(DType.uint32, [1], device=cpu),
        TensorType(DType.uint32, [1], device=cpu),
        TensorType(DType.int64, [4], device=cpu),
    ]
    if with_record:
        input_types.append(TensorType(DType.uint32, [SPLIT_SIZE], device=cpu))
    kv_params = MHAKVCacheParams(
        dtype=kv_dtype,
        n_kv_heads=NUM_KV_HEADS,
        head_dim=head_dim,
        num_layers=1,
        page_size=PAGE_SIZE,
        devices=[gpu],
    )
    sliding = shape.local_window_size > 0
    with Graph(
        f"mha_mixed_split_{with_record}_{kv_dtype}_{head_dim}_{sliding}",
        input_types=input_types,
    ) as g:
        q, offsets, blocks, lengths, lut, max_q, max_cache, metadata = g.inputs[
            :8
        ]
        assert isinstance(blocks, BufferValue)
        cache = PagedCacheValues(
            kv_blocks=blocks.buffer,
            page_stride=packed_page_stride(blocks.buffer),
            cache_lengths=lengths.tensor,
            lookup_table=lut.tensor,
            max_prompt_length=max_q.tensor,
            max_cache_length=max_cache.tensor,
            attention_dispatch_metadata=metadata.tensor,
        )
        if with_record:
            cache = replace(cache, mixed_step_split=g.inputs[8].tensor)
        out = flash_attention_ragged(
            kv_params,
            input=q.tensor,
            input_row_offsets=offsets.tensor,
            kv_collection=cache,
            layer_idx=ops.constant(0, DType.uint32, cpu),
            mask_variant=MHAMaskVariant.SLIDING_WINDOW_CAUSAL_MASK
            if sliding
            else MHAMaskVariant.CAUSAL_MASK,
            scale=1.0 / math.sqrt(head_dim),
            local_window_size=shape.local_window_size,
            output_dtype=DType.bfloat16,
        )
        g.output(out)
    return session.load(g)


class Harness:
    def __init__(self, shape: Shape) -> None:
        self.device = Accelerator()
        self.head_dim = shape.head_dim
        self.torch_dtype = TORCH_DTYPES[shape.kv_dtype]
        session = InferenceSession(devices=[self.device])
        self.split_model = _build(session, True, shape)
        self.plain_model = _build(session, False, shape)
        generator = torch.Generator(device="cuda").manual_seed(0)
        # One page at a time keeps the float32 staging small.
        self.blocks = torch.empty(
            NUM_PAGES,
            2,
            1,
            PAGE_SIZE,
            NUM_KV_HEADS,
            self.head_dim,
            dtype=self.torch_dtype,
            device="cuda",
        )
        for page in range(NUM_PAGES):
            self.blocks[page] = (
                torch.randn(
                    2,
                    1,
                    PAGE_SIZE,
                    NUM_KV_HEADS,
                    self.head_dim,
                    generator=generator,
                    device="cuda",
                )
                * 0.5
            ).to(self.torch_dtype)
        self.generator = generator

    def _inputs(
        self,
        rows: Sequence[Row],
        pages: Sequence[Sequence[int]],
        q: torch.Tensor,
    ) -> list[Buffer]:
        widths = [r.width for r in rows]
        offsets = np.zeros(len(rows) + 1, dtype=np.uint32)
        offsets[1:] = np.cumsum(widths)
        lut = np.full(
            (len(rows), padded_lut_cols(max(len(p) for p in pages))),
            NUM_PAGES,
            dtype=np.uint32,
        )
        for i, row_pages in enumerate(pages):
            lut[i, : len(row_pages)] = row_pages
        max_q = max(widths)
        max_cache = max(r.valid_length for r in rows)
        partitions = mha_decode_num_partitions(
            len(rows), max_cache, NUM_KV_HEADS, self.device
        )
        metadata = np.array(
            [len(rows), max_q, partitions, max_cache], dtype=np.int64
        )

        def gpu(array: np.ndarray) -> Buffer:
            return Buffer.from_dlpack(torch.from_numpy(array).cuda())

        return [
            Buffer.from_dlpack(q),
            gpu(offsets),
            Buffer.from_dlpack(self.blocks),
            gpu(np.array([r.prefix for r in rows], dtype=np.uint32)),
            gpu(lut),
            Buffer.from_numpy(np.array([max_q], dtype=np.uint32)),
            Buffer.from_numpy(np.array([max_cache], dtype=np.uint32)),
            Buffer.from_numpy(metadata),
        ]

    def step(self, rows: Sequence[Row]) -> tuple[list[list[int]], torch.Tensor]:
        """Returns each row's pages and a random query for ``rows``."""
        pages: list[list[int]] = []
        next_page = 0
        for row in rows:
            n = math.ceil(row.valid_length / PAGE_SIZE)
            pages.append(list(range(next_page, next_page + n)))
            next_page += n
        assert next_page <= NUM_PAGES
        q = (
            torch.randn(
                sum(r.width for r in rows),
                NUM_Q_HEADS,
                self.head_dim,
                generator=self.generator,
                device="cuda",
            )
            * 0.5
        ).to(self.torch_dtype)
        return pages, q

    def split(
        self,
        rows: Sequence[Row],
        pages: Sequence[Sequence[int]],
        q: torch.Tensor,
        record: np.ndarray,
    ) -> torch.Tensor:
        out = self.split_model.execute(
            *self._inputs(rows, pages, q), Buffer.from_numpy(record)
        )[0]
        return torch.from_dlpack(out)

    def plain(
        self,
        rows: Sequence[Row],
        pages: Sequence[Sequence[int]],
        q: torch.Tensor,
    ) -> torch.Tensor:
        out = self.plain_model.execute(*self._inputs(rows, pages, q))[0]
        return torch.from_dlpack(out)


@pytest.fixture(
    scope="module",
    params=[
        Shape(DType.float8_e4m3fn),
        Shape(DType.bfloat16),
        Shape(DType.bfloat16, head_dim=512),
        # Shorter than most rows' prefixes, so the window clips them.
        Shape(DType.bfloat16, local_window_size=512),
    ],
    ids=["fp8", "bf16", "bf16-depth512", "bf16-sliding"],
)
def harness(request: pytest.FixtureRequest) -> Harness:
    return Harness(request.param)


def _decode_rows(num: int) -> list[Row]:
    prefixes = np.linspace(300, 1_000, num).astype(int)
    return [Row(int(p), 1 + NUM_DRAFT_TOKENS) for p in prefixes]


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16)


def _assert_split_equals_parts(
    harness: Harness, rows: Sequence[Row], head_rows: int
) -> None:
    pages, q = harness.step(rows)
    whole = harness.split(rows, pages, q, _record(rows, head_rows))
    cut = sum(r.width for r in rows[:head_rows])
    head = harness.plain(rows[:head_rows], pages[:head_rows], q[:cut])
    tail = harness.plain(
        rows[head_rows:], pages[head_rows:], q[cut:].contiguous()
    )
    assert torch.equal(_bits(whole[:cut]), _bits(head))
    assert torch.equal(_bits(whole[cut:]), _bits(tail))


@pytest.mark.skipif(not is_b100_b200(), reason="the split runs on B200")
def test_verify_rows_ahead_of_a_prompt_chunk_match_their_parts(
    harness: Harness,
) -> None:
    rows = [*_decode_rows(30), Row(2_000, 200)]
    _assert_split_equals_parts(harness, rows, 30)


@pytest.mark.skipif(not is_b100_b200(), reason="the split runs on B200")
def test_a_prompt_row_ahead_of_verify_rows_matches_its_parts(
    harness: Harness,
) -> None:
    rows = [Row(0, 900), *_decode_rows(31)]
    _assert_split_equals_parts(harness, rows, 1)


@pytest.mark.skipif(not is_b100_b200(), reason="the split runs on B200")
def test_a_zeros_record_keeps_the_single_launch(harness: Harness) -> None:
    rows = [*_decode_rows(8), Row(500, 300)]
    pages, q = harness.step(rows)
    zeros = np.zeros(SPLIT_SIZE, dtype=np.uint32)
    assert torch.equal(
        _bits(harness.split(rows, pages, q, zeros)),
        _bits(harness.plain(rows, pages, q)),
    )


@pytest.mark.skipif(not is_b100_b200(), reason="the split runs on B200")
def test_without_a_record_the_verify_rows_take_another_route(
    harness: Harness,
) -> None:
    """The control: the parts tests can only fail if the routes differ."""
    rows = [*_decode_rows(30), Row(2_000, 200)]
    pages, q = harness.step(rows)
    cut = sum(r.width for r in rows[:30])
    whole = harness.plain(rows, pages, q)
    head = harness.plain(rows[:30], pages[:30], q[:cut])
    assert not torch.equal(_bits(whole[:cut]), _bits(head))


@pytest.mark.skipif(not is_b100_b200(), reason="the split runs on B200")
@pytest.mark.parametrize(
    "fault", ["head_rows_out_of_range", "two_decode_parts"]
)
def test_a_malformed_record_keeps_the_single_launch(
    harness: Harness, fault: str
) -> None:
    rows = [*_decode_rows(30), Row(2_000, 200)]
    pages, q = harness.step(rows)
    record = _record(rows, 30)
    if fault == "head_rows_out_of_range":
        record[0] = len(rows)
    else:
        record[4] = record[2]
    assert torch.equal(
        _bits(harness.split(rows, pages, q, record)),
        _bits(harness.plain(rows, pages, q)),
    )
