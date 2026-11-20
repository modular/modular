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
"""Mixed-step split of the MiniMax-M3 MSA ops (B200).

A mixed step batches decode rows (``1 + K`` tokens each) with prompt rows.
With a split record, the indexer and attention ops cut the batch into a head
and a tail view and run the decode view on the decode routes and the other on
the prefill route, in one call. The decode rows may lead the batch or trail
it. The oracle: the split output equals each view's rows run alone, bit for
bit. A zeros or bad record keeps the unsplit prefill route, so it equals the
output of a graph that passes no record.

The shape is the per-device TP4 MiniMax-M3 shape of
``Kernels/benchmarks/msa/bench_msa_decode_path.py``; the harness also runs
4 KV heads per device, the TP1 shape, and the NVFP4 KV cache, whose packed
rows and E4M3 scales run the NVFP4 ops.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import pytest
import torch
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.experimental.torch import max_dtype_to_torch
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    Graph,
    TensorType,
    TensorValue,
    ops,
)
from max.nn._graph_ops import (
    _kv_cache_row_offsets_ragged,
    msa_sparse_attention_ragged,
    msa_sparse_indexer,
)
from max.nn.kv_cache import (
    KVCacheQuantizationConfig,
    MSAKVCacheParams,
    PagedCacheValues,
    packed_page_stride,
    padded_lut_cols,
)
from max.nn.kv_cache.input_types import MIXED_STEP_SPLIT_SIZE
from test_common.graph_utils import is_b100_b200

HEAD_DIM = 128
# Query heads per KV head; the total query heads scale with the head count.
NUM_Q_HEADS = 16
PAGE_SIZE = 128
TOPK = 16
INIT_BLOCKS = 0
LOCAL_BLOCKS = 1
NUM_DRAFT_TOKENS = 3
MAX_SPEC_DRAFT = 8
SCALE = 1.0 / math.sqrt(HEAD_DIM)
# Persistent scratch rows of a 16-request replica at 1 + K tokens per row.
SCRATCH_ROWS = 16 * (NUM_DRAFT_TOKENS + 1)
MAX_CONTEXT = 32 * 1024
SCRATCH_BLOCKS = MAX_CONTEXT // PAGE_SIZE
# Packed E2M1 pairs with E4M3 scales, one scale per 16 logical elements.
NVFP4 = KVCacheQuantizationConfig(
    scale_dtype=DType.float8_e4m3fn, quantization_granularity=16
)


@dataclass(frozen=True)
class Row:
    """One request of a step: cached tokens before it, new tokens in it."""

    prefix: int
    width: int
    # Tokens the step writes past its query (Mach adds the drafter block).
    written_after: int = NUM_DRAFT_TOKENS

    @property
    def valid_length(self) -> int:
        # The host upper bound the caches use for `max_cache_length`.
        return self.prefix + self.width + self.written_after


def _decode_rows(num: int, width: int) -> list[Row]:
    prefixes = np.linspace(2_000, 20_000, num).astype(int)
    return [Row(int(p), width) for p in prefixes]


def _split_record(rows: Sequence[Row], head_rows: int) -> np.ndarray:
    """Builds the record the host rule fills for a cut after ``head_rows``."""
    head, tail = rows[:head_rows], rows[head_rows:]
    head_is_decode = max(r.width for r in head) <= MAX_SPEC_DRAFT
    prefill = tail if head_is_decode else head

    def bound(view: Sequence[Row]) -> int:
        return max(r.valid_length for r in view)

    return np.array(
        [
            head_rows,
            sum(r.width for r in head),
            max(r.width for r in head),
            bound(head),
            max(r.width for r in tail),
            bound(tail),
            sum(r.valid_length for r in prefill),
        ],
        dtype=np.uint32,
    )


def _cache_types(
    kv_components: int,
    dtype: DType,
    heads: int,
    quant: KVCacheQuantizationConfig | None,
) -> list[BufferType | TensorType]:
    gpu = DeviceRef.GPU()
    # A packed row holds two elements per byte; the scales cover groups of
    # the logical width.
    row_elems = HEAD_DIM // 2 if quant else HEAD_DIM
    blocks = BufferType(
        dtype,
        shape=[
            "num_pages",
            kv_components,
            1,
            PAGE_SIZE,
            heads,
            row_elems,
        ],
        device=gpu,
    )
    types: list[BufferType | TensorType] = [
        blocks,
        TensorType(DType.uint32, shape=["batch"], device=gpu),
        TensorType(DType.uint32, shape=["batch", "lut_cols"], device=gpu),
        TensorType(DType.uint32, shape=[1], device=DeviceRef.CPU()),
        TensorType(DType.uint32, shape=[1], device=DeviceRef.CPU()),
        TensorType(DType.int64, shape=[2], device=gpu),
    ]
    if quant is None:
        return types
    return [
        *types,
        BufferType(
            quant.scale_dtype,
            shape=[
                "num_pages",
                kv_components,
                1,
                PAGE_SIZE,
                heads,
                HEAD_DIM // quant.quantization_granularity,
            ],
            device=gpu,
        ),
        # The scales share the values' block-id space, so they read the same
        # lookup table shape.
        TensorType(DType.uint32, shape=["batch", "lut_cols"], device=gpu),
    ]


def _symbolic_cache(values: Sequence[object]) -> PagedCacheValues:
    blocks, lengths, lookup, max_prompt, max_cache, metadata = values[:6]
    assert isinstance(blocks, BufferValue)
    assert isinstance(lengths, TensorValue)
    assert isinstance(lookup, TensorValue)
    assert isinstance(max_prompt, TensorValue)
    assert isinstance(max_cache, TensorValue)
    assert isinstance(metadata, TensorValue)
    paged = PagedCacheValues(
        kv_blocks=blocks.buffer,
        page_stride=packed_page_stride(blocks.buffer),
        cache_lengths=lengths.tensor,
        lookup_table=lookup.tensor,
        max_prompt_length=max_prompt.tensor,
        max_cache_length=max_cache.tensor,
        attention_dispatch_metadata=metadata.tensor,
    )
    if len(values) == 6:
        return paged
    scales, scales_lut = values[6:]
    assert isinstance(scales, BufferValue)
    assert isinstance(scales_lut, TensorValue)
    return replace(
        paged,
        kv_scales=scales.buffer,
        scales_page_stride=packed_page_stride(scales.buffer),
        scales_lookup_table=scales_lut.tensor,
    )


def _build_model(
    session: InferenceSession,
    kv_dtype: DType,
    heads: int,
    quant: KVCacheQuantizationConfig | None,
    with_split_input: bool,
) -> Model:
    """Indexer into attention, as one sparse MiniMax-M3 layer runs them."""
    gpu = DeviceRef.GPU()
    # An NVFP4 cache dequantizes against fp8 queries; every other cache
    # scores a bf16 index query and a same-dtype attention query.
    index_q_dtype = DType.float8_e4m3fn if quant else DType.bfloat16
    attention_q_dtype = DType.float8_e4m3fn if quant else kv_dtype
    # The index-K cache packs like the main one under NVFP4.
    index_kv_dtype = kv_dtype if quant else DType.bfloat16
    index_types = _cache_types(1, index_kv_dtype, 1, quant)
    main_types = _cache_types(2, kv_dtype, heads, quant)
    input_types: list[BufferType | TensorType] = [
        TensorType(index_q_dtype, ["total_q", heads, HEAD_DIM], device=gpu),
        TensorType(
            attention_q_dtype,
            ["total_q", NUM_Q_HEADS * heads, HEAD_DIM],
            device=gpu,
        ),
        TensorType(DType.uint32, ["batch_p1"], device=gpu),
        TensorType(DType.uint32, [1], device=DeviceRef.CPU()),
        *index_types,
        *main_types,
        BufferType(
            DType.float32, [heads, "scratch_rows", SCRATCH_BLOCKS], device=gpu
        ),
    ]
    if with_split_input:
        input_types.append(
            TensorType(
                DType.uint32, [MIXED_STEP_SPLIT_SIZE], device=DeviceRef.CPU()
            )
        )
    kv_params = MSAKVCacheParams(
        dtype=kv_dtype,
        n_kv_heads=heads,
        head_dim=HEAD_DIM,
        num_layers=1,
        devices=[gpu],
        page_size=PAGE_SIZE,
        kvcache_quant_config=quant,
    )
    # The index-K cache is MLA with one latent head at every layout, so the
    # indexer gets its own one-head params, as the model's indexer does.
    index_kv_params = replace(kv_params, n_kv_heads=1)
    name = f"msa_mixed_split_{kv_dtype}_{heads}h_{with_split_input}"
    with Graph(name, input_types=input_types) as graph:
        values = list(graph.inputs)
        index_cache = _symbolic_cache(values[4 : 4 + len(index_types)])
        main_start = 4 + len(index_types)
        main_cache = _symbolic_cache(
            values[main_start : main_start + len(main_types)]
        )
        scratch = values[main_start + len(main_types)]
        split_args = (
            values[main_start + len(main_types) + 1].tensor
            if with_split_input
            else None
        )
        index_cache = replace(index_cache, mixed_step_split=split_args)
        main_cache = replace(main_cache, mixed_step_split=split_args)
        row_offsets = values[2].tensor
        layer_idx = ops.constant(0, DType.uint32, DeviceRef.CPU())
        block_indices = msa_sparse_indexer(
            index_kv_params,
            index_q=values[0].tensor,
            input_row_offsets=row_offsets,
            prefix_lens=index_cache.cache_lengths,
            index_kv_collection=index_cache,
            layer_idx=layer_idx,
            score_scratch=scratch.buffer,
            num_index_heads=heads,
            idx_head_dim=HEAD_DIM,
            block_size=PAGE_SIZE,
            topk=TOPK,
            init_blocks=INIT_BLOCKS,
            local_blocks=LOCAL_BLOCKS,
            scale=SCALE,
        )
        output = msa_sparse_attention_ragged(
            kv_params,
            input=values[1].tensor,
            input_row_offsets=row_offsets,
            cache_row_offsets=_kv_cache_row_offsets_ragged(
                row_offsets, main_cache
            ),
            total_context_length=values[3].tensor,
            kv_collection=main_cache,
            layer_idx=layer_idx,
            block_indices=block_indices,
            group=NUM_Q_HEADS,
            topk=TOPK,
            sparse_block_size=PAGE_SIZE,
            scale=SCALE,
        )
        graph.output(block_indices, output)
    return session.load(graph)


@dataclass
class Step:
    """Device tensors of one step over a row subset of the shared pool."""

    index_q: torch.Tensor
    attention_q: torch.Tensor
    rows: list[Row]
    pages: list[list[int]]
    # The staged `max_cache_length`; None stages the rows' largest.
    max_cache_length: int | None = None

    def inputs(
        self,
        index_blocks: torch.Tensor,
        index_scales: torch.Tensor | None,
        main_blocks: torch.Tensor,
        main_scales: torch.Tensor | None,
        scratch: torch.Tensor,
    ) -> list[Buffer]:
        widths = [r.width for r in self.rows]
        row_offsets = np.zeros(len(self.rows) + 1, dtype=np.uint32)
        row_offsets[1:] = np.cumsum(widths)
        lut_cols = padded_lut_cols(max(len(p) for p in self.pages))
        lut = np.full(
            (len(self.rows), lut_cols), index_blocks.shape[0], dtype=np.uint32
        )
        for i, row_pages in enumerate(self.pages):
            lut[i, : len(row_pages)] = row_pages
        prefixes = np.array([r.prefix for r in self.rows], dtype=np.uint32)
        max_prompt = np.array([max(widths)], dtype=np.uint32)
        max_cache = np.array(
            [self.max_cache_length or max(r.valid_length for r in self.rows)],
            dtype=np.uint32,
        )
        total_context = np.array(
            [sum(r.valid_length for r in self.rows)], dtype=np.uint32
        )
        metadata = torch.tensor(
            [len(self.rows), int(max_cache[0])], dtype=torch.int64
        ).cuda()

        def gpu(array: np.ndarray) -> Buffer:
            return Buffer.from_dlpack(torch.from_numpy(array).cuda())

        def cpu(array: np.ndarray) -> Buffer:
            return Buffer.from_dlpack(torch.from_numpy(array))

        def cache(
            blocks: torch.Tensor, scales: torch.Tensor | None
        ) -> list[Buffer]:
            # The scales share the values' block-id space, so they read the
            # same table.
            entries = [
                Buffer.from_dlpack(blocks),
                gpu(prefixes),
                gpu(lut),
                cpu(max_prompt),
                cpu(max_cache),
                Buffer.from_dlpack(metadata),
            ]
            if scales is not None:
                entries += [Buffer.from_dlpack(scales), gpu(lut)]
            return entries

        return [
            Buffer.from_dlpack(self.index_q),
            Buffer.from_dlpack(self.attention_q),
            gpu(row_offsets),
            cpu(total_context),
            *cache(index_blocks, index_scales),
            *cache(main_blocks, main_scales),
            Buffer.from_dlpack(scratch),
        ]

    def subset(self, start: int, stop: int) -> Step:
        token_start = sum(r.width for r in self.rows[:start])
        token_stop = sum(r.width for r in self.rows[:stop])
        return Step(
            index_q=self.index_q[token_start:token_stop].contiguous(),
            attention_q=self.attention_q[token_start:token_stop].contiguous(),
            rows=self.rows[start:stop],
            pages=self.pages[start:stop],
            max_cache_length=self.max_cache_length,
        )


class Harness:
    """Shared KV pool, compiled graphs and the run helpers of one KV dtype."""

    def __init__(
        self,
        session: InferenceSession,
        kv_dtype: DType,
        heads: int,
        quant: KVCacheQuantizationConfig | None,
    ) -> None:
        self.kv_dtype = kv_dtype
        self.heads = heads
        self.quant = quant
        self.split_model = _build_model(session, kv_dtype, heads, quant, True)
        self.plain_model = _build_model(session, kv_dtype, heads, quant, False)
        self.num_pages = 1_024
        generator = torch.Generator(device="cuda").manual_seed(0)
        if quant is None:
            self.index_blocks = torch.randn(
                self.num_pages,
                1,
                1,
                PAGE_SIZE,
                1,
                HEAD_DIM,
                dtype=torch.bfloat16,
                device="cuda",
                generator=generator,
            )
            main = torch.randn(
                self.num_pages,
                2,
                1,
                PAGE_SIZE,
                heads,
                HEAD_DIM,
                dtype=torch.bfloat16,
                device="cuda",
                generator=generator,
            )
            self.main_blocks = main.to(max_dtype_to_torch(self.kv_dtype))
            self.index_scales = None
            self.main_scales = None
        else:
            # Random packed bytes dequantize to arbitrary E2M1 pairs, which
            # is all a bit-identity oracle needs.
            self.index_blocks = torch.randint(
                0,
                256,
                (self.num_pages, 1, 1, PAGE_SIZE, 1, HEAD_DIM // 2),
                dtype=torch.uint8,
                device="cuda",
                generator=generator,
            )
            self.main_blocks = torch.randint(
                0,
                256,
                (self.num_pages, 2, 1, PAGE_SIZE, heads, HEAD_DIM // 2),
                dtype=torch.uint8,
                device="cuda",
                generator=generator,
            )
            self.index_scales = self._scales(1, 1, generator)
            self.main_scales = self._scales(2, heads, generator)
        self.generator = generator

    def _scales(
        self,
        kv_components: int,
        heads: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        assert self.quant is not None
        # Finite scales keep every dequantized product finite.
        return (
            torch.rand(
                (
                    self.num_pages,
                    kv_components,
                    1,
                    PAGE_SIZE,
                    heads,
                    HEAD_DIM // self.quant.quantization_granularity,
                ),
                device="cuda",
                generator=generator,
            )
            * 2
            + 0.5
        ).to(max_dtype_to_torch(self.quant.scale_dtype))

    def step(
        self, rows: Sequence[Row], max_cache_length: int | None = None
    ) -> Step:
        total_q = sum(r.width for r in rows)
        index_q = torch.randn(
            total_q,
            self.heads,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device="cuda",
            generator=self.generator,
        )
        attention_q = torch.randn(
            total_q,
            NUM_Q_HEADS * self.heads,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device="cuda",
            generator=self.generator,
        )
        if self.quant is not None:
            # An NVFP4 cache dequantizes against fp8 queries.
            index_q = index_q.to(torch.float8_e4m3fn)
            attention_q = attention_q.to(torch.float8_e4m3fn)
        else:
            attention_q = attention_q.to(max_dtype_to_torch(self.kv_dtype))
        pages: list[list[int]] = []
        next_page = 0
        for r in rows:
            count = math.ceil(r.valid_length / PAGE_SIZE)
            pages.append(list(range(next_page, next_page + count)))
            next_page += count
        assert next_page <= self.num_pages, "grow the page pool"
        return Step(index_q, attention_q, list(rows), pages, max_cache_length)

    def scratch(self, rows: int = SCRATCH_ROWS) -> torch.Tensor:
        return torch.zeros(
            self.heads, rows, SCRATCH_BLOCKS, dtype=torch.float32, device="cuda"
        )

    def run_split(
        self,
        step: Step,
        record: np.ndarray,
        scratch: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        buffers = step.inputs(
            self.index_blocks,
            self.index_scales,
            self.main_blocks,
            self.main_scales,
            self.scratch() if scratch is None else scratch,
        )
        buffers.append(Buffer.from_dlpack(torch.from_numpy(record)))
        return self._outputs(self.split_model.execute(*buffers))

    def run_plain(
        self, step: Step, scratch: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        buffers = step.inputs(
            self.index_blocks,
            self.index_scales,
            self.main_blocks,
            self.main_scales,
            self.scratch() if scratch is None else scratch,
        )
        return self._outputs(self.plain_model.execute(*buffers))

    @staticmethod
    def _outputs(
        results: Sequence[object],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        indices, output = results
        assert isinstance(indices, Buffer)
        assert isinstance(output, Buffer)
        return (
            torch.from_dlpack(indices).cpu().clone(),
            torch.from_dlpack(output).cpu().clone(),
        )


# Every harness the test targets pick from, by test id.
HARNESSES: dict[str, tuple[DType, int, KVCacheQuantizationConfig | None]] = {
    "bf16_1h": (DType.bfloat16, 1, None),
    "bf16_4h": (DType.bfloat16, 4, None),
    "fp8_1h": (DType.float8_e4m3fn, 1, None),
    "fp8_4h": (DType.float8_e4m3fn, 4, None),
    "nvfp4_1h": (DType.uint8, 1, NVFP4),
    "nvfp4_4h": (DType.uint8, 4, NVFP4),
}


def harness_fixture(*ids: str) -> object:
    """Returns a class-scoped ``harness`` fixture over the given harness ids."""

    @pytest.fixture(
        scope="class", params=[HARNESSES[i] for i in ids], ids=list(ids)
    )
    def harness(
        self: object,
        session: InferenceSession,
        request: pytest.FixtureRequest,
    ) -> Harness:
        kv_dtype, heads, quant = request.param
        return Harness(session, kv_dtype, heads, quant)

    return harness


def _assert_bitwise_equal(
    actual: torch.Tensor, expected: torch.Tensor, what: str
) -> None:
    assert actual.shape == expected.shape, what
    same = actual.view(torch.uint8) == expected.view(torch.uint8)
    if not bool(same.all()):
        mismatched = int((~same).sum())
        pytest.fail(f"{what}: {mismatched} bytes differ")


def _assert_split_equals_parts(
    harness: Harness,
    rows: Sequence[Row],
    head_rows: int,
    max_cache_length: int | None = None,
) -> None:
    step = harness.step(rows, max_cache_length)
    head_tokens = sum(r.width for r in rows[:head_rows])
    split_indices, split_output = harness.run_split(
        step, _split_record(rows, head_rows)
    )
    head_indices, head_output = harness.run_plain(step.subset(0, head_rows))
    tail_indices, tail_output = harness.run_plain(
        step.subset(head_rows, len(rows))
    )
    _assert_bitwise_equal(
        split_indices[:, :head_tokens], head_indices, "head-view block ids"
    )
    _assert_bitwise_equal(
        split_indices[:, head_tokens:], tail_indices, "tail-view block ids"
    )
    _assert_bitwise_equal(
        split_output[:head_tokens], head_output, "head-view output"
    )
    _assert_bitwise_equal(
        split_output[head_tokens:], tail_output, "tail-view output"
    )


def _assert_record_keeps_unsplit_route(
    harness: Harness,
    rows: Sequence[Row],
    record: np.ndarray,
    scratch_rows: int = SCRATCH_ROWS,
) -> None:
    step = harness.step(rows)
    split_indices, split_output = harness.run_split(
        step, record, harness.scratch(scratch_rows)
    )
    plain_indices, plain_output = harness.run_plain(
        step, harness.scratch(scratch_rows)
    )
    _assert_bitwise_equal(split_indices, plain_indices, "block ids")
    _assert_bitwise_equal(split_output, plain_output, "output")


VERIFY_WIDTH = 1 + NUM_DRAFT_TOKENS
MIXED_ROWS = [*_decode_rows(8, VERIFY_WIDTH), Row(1_000, 600)]
SINGLE_TOKEN_ROWS = [*_decode_rows(8, 1), Row(1_000, 600)]


# A verify row of seven drafts plus its anchor: the widest query the
# speculative-decode routes take.
WIDEST_VERIFY = MAX_SPEC_DRAFT


# Mach's DSpark export: K = 7 drafts plus a 7-token drafter block written per
# row; the ceiling is the leaf's staged `max_cache_length`, not the record's.
MACH_BLOCK = 7
MACH_CEILING = 16_384 + 7 + 1


def _bad_records() -> dict[str, np.ndarray]:
    good = _split_record(MIXED_ROWS, 8)
    cases: dict[str, np.ndarray] = {}

    def variant(name: str, slot: int, value: int) -> None:
        record = good.copy()
        record[slot] = value
        cases[name] = record

    variant("head_holds_every_row", 0, len(MIXED_ROWS))
    variant("two_decode_views", 4, MAX_SPEC_DRAFT)
    variant("head_tokens_above_rows_times_width", 1, 8 * VERIFY_WIDTH + 1)
    variant("tail_width_above_batch_max", 4, 601)
    variant("head_cache_above_batch_max", 3, 1 << 20)
    variant("tail_cache_above_batch_max", 5, 1 << 20)
    variant("prefill_context_below_prefill_tokens", 6, 599)
    variant("prefill_context_above_rows_times_cache", 6, 1 << 30)
    return cases


class MixedSplitCases:
    """The split tests; each test file subclasses this with its harnesses."""

    @pytest.fixture(scope="class")
    def session(self) -> InferenceSession:
        if not is_b100_b200():
            pytest.skip("the MSA split runs on NVIDIA SM100 only")
        return InferenceSession(devices=[Accelerator()])

    def test_verify_rows_and_a_prompt_chunk_match_their_parts(
        self,
        harness: Harness,
    ) -> None:
        _assert_split_equals_parts(harness, MIXED_ROWS, head_rows=8)

    def test_one_trailing_decode_row_matches_its_parts(
        self, harness: Harness
    ) -> None:
        rows = [Row(1_000, 600), Row(0, 300), Row(9_000, VERIFY_WIDTH)]
        _assert_split_equals_parts(harness, rows, head_rows=2)

    def test_single_token_decode_rows_match_their_parts(
        self, harness: Harness
    ) -> None:
        _assert_split_equals_parts(harness, SINGLE_TOKEN_ROWS, head_rows=8)

    def test_without_a_record_the_verify_rows_take_another_route(
        self,
        harness: Harness,
    ) -> None:
        # Control for the tests above: the unsplit route gives the verify rows
        # other bits, so equality with their parts shows that the split ran.
        step = harness.step(MIXED_ROWS)
        head_tokens = 8 * VERIFY_WIDTH
        plain_indices, plain_output = harness.run_plain(step)
        head_indices, head_output = harness.run_plain(step.subset(0, 8))
        assert not (
            torch.equal(plain_indices[:, :head_tokens], head_indices)
            and torch.equal(
                plain_output[:head_tokens].view(torch.uint8),
                head_output.view(torch.uint8),
            )
        )

    def test_widest_verify_rows_after_the_prompt_rows_match_their_parts(
        self,
        harness: Harness,
    ) -> None:
        rows = [Row(1_000, 600), *_decode_rows(8, WIDEST_VERIFY)]
        _assert_split_equals_parts(harness, rows, head_rows=1)

    def test_a_row_past_the_decode_ceiling_rides_the_prefill_view(
        self,
        harness: Harness,
    ) -> None:
        # A one-token chunk is as wide as a verify row; the first width past
        # the decode ceiling sends the tail down the prefill route.
        rows = [
            *_decode_rows(6, WIDEST_VERIFY),
            Row(500, WIDEST_VERIFY),
            Row(700, WIDEST_VERIFY + 1),
        ]
        _assert_split_equals_parts(harness, rows, head_rows=7)

    def test_a_mach_shaped_record_matches_its_parts(
        self, harness: Harness
    ) -> None:
        # Mach order: a prompt chunk with its MAGIC drafts, then verify rows.
        prefixes = np.linspace(2_000, 12_000, 8).astype(int)
        rows = [
            Row(1_000, 600 + 7, MACH_BLOCK),
            *(Row(int(p), WIDEST_VERIFY, MACH_BLOCK) for p in prefixes),
        ]
        _assert_split_equals_parts(harness, rows, 1, MACH_CEILING)

    def test_zeros_record_keeps_the_unsplit_route(
        self, harness: Harness
    ) -> None:
        _assert_record_keeps_unsplit_route(
            harness,
            MIXED_ROWS,
            np.zeros(MIXED_STEP_SPLIT_SIZE, dtype=np.uint32),
        )

    @pytest.mark.parametrize("name", sorted(_bad_records()))
    def test_bad_record_keeps_the_unsplit_route(
        self, harness: Harness, name: str
    ) -> None:
        _assert_record_keeps_unsplit_route(
            harness, MIXED_ROWS, _bad_records()[name]
        )

    def test_scratch_too_small_for_the_decode_rows_keeps_the_unsplit_indexer(
        self,
        harness: Harness,
    ) -> None:
        # The single-token decode route raises on a too-small scratch, so the
        # split refuses and the unsplit route runs.
        step = harness.step(SINGLE_TOKEN_ROWS)
        split_indices, _ = harness.run_split(
            step, _split_record(SINGLE_TOKEN_ROWS, 8), harness.scratch(4)
        )
        plain_indices, _ = harness.run_plain(step, harness.scratch(16))
        _assert_bitwise_equal(split_indices, plain_indices, "block ids")
