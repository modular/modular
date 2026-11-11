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
"""A mixed batch through the MLA indexer's row-remapped ops equals its parts.

A decode row and a prefill row go through the indexer in both row orders, and
each row must select what the same row selects alone: on the split route a
view's launch is the standalone launch, so every token picks the same logical
positions and fills the same number of slots, and the remap arms also pick the
same rows of the MLA value pool and of its scale pool. The top-k is unordered,
so a compare is on sorted sets. A zeros record and a record the split rules
refuse both keep the whole-batch launch, so their outputs agree.

The arms are the ``.paged`` registration, which writes the logical positions
only, and the three the sparse path uses: ``.remap`` over an FP8 MLA cache,
``.remap.nvfp4`` over an NVFP4 one, whose rows carry each selection's
scale-pool row after its value row, and ``.remap`` with ``write_logical``
off, which writes only the physical rows.

The score of a key position is that position's f32 key scale by construction,
so a row's selections are its last ``top_k`` positions and no tie decides
them; ``_assert_no_ties`` checks that on the same host key tiles.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pytest
import torch
from _mla_sparse_test_utils import paged_kv_from_flat_graph_inputs
from max import tree
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn._graph_ops import (
    mla_fp8_index_top_k,
    mla_fp8_index_top_k_physical,
    mla_fp8_index_top_k_with_physical,
)
from max.nn.kv_cache import (
    MIXED_STEP_SPLIT_SIZE,
    KVCacheInputsPerDevice,
    KVCacheQuantizationConfig,
    MLAKVCacheParams,
    padded_lut_cols,
)
from max.pipelines.kv_cache.paged_kv_cache._mixed_step_split import (
    mixed_step_decode_width,
    mixed_step_split_record,
)
from test_common.graph_utils import is_b100_b200
from test_common.simple_kv_cache import (
    block_ids_for_batch,
    paged_kv_cache_inputs,
)
from torch.utils.dlpack import from_dlpack

NUM_HEADS = 8
INDEX_HEAD_DIM = 128
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
PAGE_SIZE = 128
TOPK = 64

RowKind = Literal["decode", "prefill"]

# Each key position's score is its key scale, and one binade per this many
# positions leaves every score more than a bf16 ulp from its neighbour.
SCORES_PER_BINADE = 64

# The record slots `read_mixed_split` reads.
_HEAD_ROWS_SLOT = 0
_HEAD_TOKENS_SLOT = 1


@dataclass(frozen=True)
class Row:
    """One request of the step: cached tokens, then the tokens it adds."""

    prefix: int
    tokens: int

    @property
    def valid_length(self) -> int:
        """Returns the tokens the row's cache bound covers."""
        return self.prefix + self.tokens


POOL: dict[RowKind, Row] = {
    "decode": Row(2_300, 1),
    "prefill": Row(0, 600),
}
POOL_ORDER: tuple[RowKind, ...] = ("decode", "prefill")
_PAGES: dict[RowKind, np.ndarray] = {
    row: np.array(pages)
    for row, pages in zip(
        POOL_ORDER,
        block_ids_for_batch(
            [POOL[row].valid_length for row in POOL_ORDER], PAGE_SIZE
        ),
        strict=True,
    )
}
# The scale pool pages its blocks apart from the values', so a scale row taken
# from the value table shows up as another position's scale row.
_SCALE_PAGES: dict[RowKind, np.ndarray] = {
    row: np.roll(pages, 1) for row, pages in _PAGES.items()
}
_TOTAL_PAGES = sum(len(pages) for pages in _PAGES.values())
_LUT_COLUMNS = padded_lut_cols(
    max(-(-POOL[row].valid_length // PAGE_SIZE) for row in POOL_ORDER)
)


@dataclass(frozen=True)
class Arm:
    """One indexer registration and its MLA cache dtype, `None` for none."""

    name: str
    mla_dtype: DType | None
    writes_logical: bool = True

    @property
    def writes_physical(self) -> bool:
        """Returns whether the op writes each selection's MLA row too."""
        return self.mla_dtype is not None


ARMS: tuple[Arm, ...] = (
    Arm("remap", DType.float8_e4m3fn),
    Arm("remap_nvfp4", DType.uint8),
    Arm("physical", DType.float8_e4m3fn, writes_logical=False),
    Arm("paged", None),
)


def _indexer_params() -> MLAKVCacheParams:
    """Builds the FP8 indexer cache the op scores against."""
    return MLAKVCacheParams(
        dtype=DType.float8_e4m3fn,
        head_dim=INDEX_HEAD_DIM,
        num_layers=1,
        page_size=PAGE_SIZE,
        devices=[DeviceRef.GPU()],
        num_q_heads=NUM_HEADS,
        kvcache_quant_config=KVCacheQuantizationConfig(
            scale_dtype=DType.float32,
            quantization_granularity=INDEX_HEAD_DIM,
        ),
    )


def _mla_params(arm: Arm) -> MLAKVCacheParams:
    """Builds the MLA cache the arm's physical rows point into."""
    dtype = arm.mla_dtype
    assert dtype is not None, "only a physical arm has an MLA cache"
    return MLAKVCacheParams(
        dtype=dtype,
        head_dim=KV_LORA_RANK + QK_ROPE_HEAD_DIM,
        num_layers=1,
        page_size=PAGE_SIZE,
        devices=[DeviceRef.GPU()],
        num_q_heads=NUM_HEADS,
        kvcache_quant_config=KVCacheQuantizationConfig(
            scale_dtype=DType.float8_e4m3fn,
            quantization_granularity=16,
            fp8_tail_dim=QK_ROPE_HEAD_DIM,
        )
        if dtype is DType.uint8
        else None,
    )


def _graph(arm: Arm) -> Graph:
    """Builds the indexer op over the arm's caches, its record an input."""
    gpu = DeviceRef.GPU()
    indexer = _indexer_params()
    indexer_leaves = tree.leaves(indexer.get_symbolic_inputs()[0])
    mla = _mla_params(arm) if arm.writes_physical else None
    mla_leaves = (
        [] if mla is None else tree.leaves(mla.get_symbolic_inputs()[0])
    )
    input_types = [
        TensorType(
            DType.float8_e4m3fn,
            ["total_q", NUM_HEADS, INDEX_HEAD_DIM],
            gpu,
        ),
        TensorType(DType.float32, ["total_q", NUM_HEADS], gpu),
        TensorType(DType.uint32, ["rows_p1"], gpu),
        *indexer_leaves,
        *mla_leaves,
    ]
    name = f"mla_indexer_mixed_split_{arm.name}"
    with Graph(name, input_types=input_types) as g:
        q, q_s, row_offsets = (value.tensor for value in g.inputs[:3])
        flat = list(g.inputs[3:])
        index_cache = paged_kv_from_flat_graph_inputs(
            indexer, flat[: len(indexer_leaves)]
        )
        layer = ops.constant(0, DType.uint32, device=DeviceRef.CPU())
        if mla is None:
            g.output(
                mla_fp8_index_top_k(
                    q,
                    q_s,
                    row_offsets,
                    index_cache,
                    layer,
                    TOPK,
                    INDEX_HEAD_DIM,
                    kpool=1,
                )
            )
        else:
            mla_cache = paged_kv_from_flat_graph_inputs(
                mla, flat[len(indexer_leaves) :]
            )
            if arm.writes_logical:
                g.output(
                    *mla_fp8_index_top_k_with_physical(
                        q,
                        q_s,
                        row_offsets,
                        index_cache,
                        mla_cache,
                        layer,
                        TOPK,
                        INDEX_HEAD_DIM,
                    )
                )
            else:
                g.output(
                    mla_fp8_index_top_k_physical(
                        q,
                        q_s,
                        row_offsets,
                        index_cache,
                        mla_cache,
                        layer,
                        TOPK,
                        INDEX_HEAD_DIM,
                    )
                )
    return g


def _profile(length: int) -> np.ndarray:
    """Returns each key position's score, graded so no two scores tie."""
    return np.exp2(
        np.arange(length, dtype=np.float64) / SCORES_PER_BINADE
    ).astype(np.float32)


def _row_keys(row: RowKind) -> torch.Tensor:
    """Returns a row's cached keys, one ``e_0`` unit vector per position."""
    keys = torch.zeros(
        (POOL[row].valid_length, INDEX_HEAD_DIM), dtype=torch.float8_e4m3fn
    )
    keys[:, 0] = 1.0
    return keys


def _row_queries(row: RowKind) -> torch.Tensor:
    """Returns a row's queries, one unit vector at the first head."""
    q = torch.zeros(
        (POOL[row].tokens, NUM_HEADS, INDEX_HEAD_DIM),
        dtype=torch.float8_e4m3fn,
    )
    q[:, 0, 0] = 1.0
    return q


def _row_query_scales(row: RowKind) -> torch.Tensor:
    """Returns a row's per-head query scales."""
    return torch.ones((POOL[row].tokens, NUM_HEADS), dtype=torch.float32)


def _index_pool(
    params: MLAKVCacheParams,
    keys: Mapping[RowKind, torch.Tensor],
    profile: Mapping[RowKind, np.ndarray],
    device: Accelerator,
) -> tuple[Buffer, Buffer]:
    """Writes each row's keys and their scores into the shared page pool."""
    blocks = torch.zeros(
        (_TOTAL_PAGES + 1, *tuple(int(d) for d in params.shape_per_block)),
        dtype=torch.float8_e4m3fn,
    )
    scale_block = tuple(int(d) for d in params.shape_per_scale_block)
    scales = torch.zeros((_TOTAL_PAGES + 1, *scale_block), dtype=torch.float32)
    assert blocks.shape[3] == scales.shape[3] == PAGE_SIZE, (
        blocks.shape,
        scales.shape,
    )
    for row in POOL_ORDER:
        length = POOL[row].valid_length
        for page, offset in zip(
            _PAGES[row], range(0, length, PAGE_SIZE), strict=True
        ):
            count = min(PAGE_SIZE, length - offset)
            blocks[page, 0, 0, :count, 0, 0] = keys[row][
                offset : offset + count, 0
            ]
            scales[page, 0, 0, :count, 0, 0] = torch.from_numpy(
                profile[row][offset : offset + count]
            )
    return (
        Buffer.from_dlpack(blocks.view(torch.uint8))
        .view(DType.float8_e4m3fn)
        .to(device),
        Buffer.from_dlpack(scales).to(device),
    )


def _assert_no_ties(
    row: RowKind,
    q: torch.Tensor,
    q_s: torch.Tensor,
    keys: torch.Tensor,
    scale: np.ndarray,
) -> None:
    """Pins that no tie decides the last selected key of this row's tokens."""
    dots = torch.einsum("thd,pd->tph", q.float(), keys.float())
    scores = (q_s.unsqueeze(1) * dots.clamp_min(0)).sum(-1)
    scores = scores * torch.from_numpy(scale)[None, :]
    for token in range(POOL[row].tokens):
        bound = POOL[row].prefix + token + 1
        if bound <= TOPK:
            continue
        top = torch.topk(scores[token, :bound], TOPK + 1).values
        assert top[-2].to(torch.bfloat16) != top[-1].to(torch.bfloat16), (
            f"{row} token {token}: the {TOPK}th score ties the next"
        )


@dataclass(frozen=True)
class Step:
    """One arm's shared page pool and the queries of its rows."""

    arm: Arm
    device: Accelerator
    q: Mapping[RowKind, torch.Tensor]
    q_s: Mapping[RowKind, torch.Tensor]
    index_blocks: Buffer
    index_scales: Buffer


def _step(arm: Arm) -> Step:
    """Builds one arm's shared page pool and its queries."""
    device = Accelerator(0)
    keys = {row: _row_keys(row) for row in POOL_ORDER}
    profile = {row: _profile(POOL[row].valid_length) for row in POOL_ORDER}
    q = {row: _row_queries(row) for row in POOL_ORDER}
    q_s = {row: _row_query_scales(row) for row in POOL_ORDER}
    for row in POOL_ORDER:
        _assert_no_ties(row, q[row], q_s[row], keys[row], profile[row])
    blocks, scales = _index_pool(_indexer_params(), keys, profile, device)
    return Step(
        arm=arm,
        device=device,
        q=q,
        q_s=q_s,
        index_blocks=blocks,
        index_scales=scales,
    )


def _table(
    rows: Sequence[RowKind],
    pages: Mapping[RowKind, np.ndarray],
    device: Accelerator,
) -> Buffer:
    """Builds a page lookup table for ``rows`` out of the shared pool."""
    table = np.full((len(rows), _LUT_COLUMNS), _TOTAL_PAGES, np.uint32)
    for index, row in enumerate(rows):
        row_pages = pages[row]
        table[index, : len(row_pages)] = row_pages
    return Buffer.from_numpy(table).to(device)


def _leaf(
    params: MLAKVCacheParams,
    rows: Sequence[RowKind],
    device: Accelerator,
) -> KVCacheInputsPerDevice[Buffer, Buffer]:
    """Builds a cache leaf for ``rows``, its value table in the shared pages."""
    leaf = paged_kv_cache_inputs(
        params,
        [POOL[row].tokens for row in rows],
        cache_lengths=[POOL[row].prefix for row in rows],
        total_num_pages=_TOTAL_PAGES,
    )
    leaf.lookup_table = _table(rows, _PAGES, device)
    return leaf


def _index_cache(
    rows: Sequence[RowKind], step: Step, device: Accelerator
) -> KVCacheInputsPerDevice[Buffer, Buffer]:
    """Builds the indexer's leaf for ``rows`` over the shared key pool."""
    leaf = _leaf(_indexer_params(), rows, device)
    leaf.kv_blocks = step.index_blocks
    leaf.kv_scales = step.index_scales
    # The scales share the values' block ids.
    leaf.scales_lookup_table = leaf.lookup_table
    return leaf


def _mla_cache(
    arm: Arm, rows: Sequence[RowKind], device: Accelerator
) -> KVCacheInputsPerDevice[Buffer, Buffer]:
    """Builds the MLA leaf for ``rows``, its two pools paged apart."""
    leaf = _leaf(_mla_params(arm), rows, device)
    if leaf.scales_lookup_table is not None:
        leaf.scales_lookup_table = _table(rows, _SCALE_PAGES, device)
    return leaf


def _record(rows: Sequence[RowKind]) -> np.ndarray:
    """Builds the split record the KV managers fill for ``rows``."""
    return mixed_step_split_record(
        [POOL[row].tokens for row in rows],
        [POOL[row].valid_length for row in rows],
        mixed_step_decode_width(0),
    )


def _refused_record(rows: Sequence[RowKind]) -> np.ndarray:
    """Returns the split record with a head view that holds no tokens."""
    record = _record(rows).copy()
    record[_HEAD_TOKENS_SLOT] = 0
    return record


def _fp8_buffer(packed: torch.Tensor, device: Accelerator) -> Buffer:
    """Returns FP8 bytes as an FP8 device buffer."""
    return Buffer.from_dlpack(packed).view(DType.float8_e4m3fn).to(device)


def _ints(values: Buffer) -> np.ndarray:
    """Returns an int32 output buffer as a host array."""
    return from_dlpack(values).cpu().numpy()


def _run(
    model: Model,
    step: Step,
    rows: Sequence[RowKind],
    record: np.ndarray,
) -> list[np.ndarray]:
    """Runs ``rows`` through the arm's op under ``record``."""
    device = step.device
    index_cache = _index_cache(rows, step, device)
    index_cache.mixed_step_split = Buffer.from_numpy(record)
    offsets = np.zeros(len(rows) + 1, dtype=np.uint32)
    offsets[1:] = np.cumsum([POOL[row].tokens for row in rows])
    inputs = [
        _fp8_buffer(
            torch.cat([step.q[row].view(torch.uint8) for row in rows]),
            device,
        ),
        Buffer.from_dlpack(torch.cat([step.q_s[row] for row in rows])).to(
            device
        ),
        Buffer.from_numpy(offsets).to(device),
        *tree.leaves(index_cache),
    ]
    if step.arm.writes_physical:
        inputs += tree.leaves(_mla_cache(step.arm, rows, device))
    return [_ints(values) for values in model.execute(*inputs)]


def _blocks(values: np.ndarray, columns: int) -> list[np.ndarray]:
    """Splits an output into ``columns``-wide blocks: values, then scales."""
    return [
        values[:, offset : offset + columns]
        for offset in range(0, values.shape[1], columns)
    ]


def _live(values: np.ndarray) -> np.ndarray:
    """Returns a token's selected slots, sorted; ``-1`` marks a dead slot."""
    return np.sort(values[values >= 0])


def _row_blocks(
    outputs: Sequence[np.ndarray],
    start: int,
    tokens: int,
) -> list[np.ndarray]:
    """Returns each output's blocks over one row range."""
    return [
        values[start : start + tokens]
        for output in outputs
        for values in _blocks(output, TOPK)
    ]


def _assert_row_matches(
    got: Sequence[np.ndarray],
    alone: Sequence[np.ndarray],
    rows: Sequence[RowKind],
    row: RowKind,
    why: str,
) -> None:
    """Pins that ``row`` in a batch selects what the row selects alone."""
    start = sum(POOL[earlier].tokens for earlier in rows[: rows.index(row)])
    tokens = POOL[row].tokens
    prefix = POOL[row].prefix
    got_blocks = _row_blocks(got, start, tokens)
    alone_blocks = _row_blocks(alone, 0, tokens)
    for block, (got_block, alone_block) in enumerate(
        zip(got_blocks, alone_blocks, strict=True)
    ):
        for token in range(tokens):
            got_live = _live(got_block[token])
            np.testing.assert_array_equal(
                got_live,
                _live(alone_block[token]),
                err_msg=f"{why}: block {block}, token {token}",
            )
            assert len(got_live) == min(TOPK, prefix + token + 1), (
                f"{why}: block {block}, token {token} fills "
                f"{len(got_live)} slots"
            )


def _assert_same_run(
    got: Sequence[np.ndarray],
    want: Sequence[np.ndarray],
    why: str,
) -> None:
    """Pins that two runs of one batch select the same sets, per token."""
    tokens = got[0].shape[0]
    for got_block, want_block in zip(
        _row_blocks(got, 0, tokens),
        _row_blocks(want, 0, tokens),
        strict=True,
    ):
        for token in range(tokens):
            np.testing.assert_array_equal(
                _live(got_block[token]),
                _live(want_block[token]),
                err_msg=f"{why}: token {token}",
            )


@pytest.fixture(scope="module")
def models() -> dict[str, Model]:
    """Loads one graph per arm, compiled by the CPU record action."""
    if not is_b100_b200():
        pytest.skip("the MLA indexer's row remap runs on NVIDIA SM100 only")
    session = InferenceSession(devices=[Accelerator()])
    return {arm.name: session.load(_graph(arm)) for arm in ARMS}


@pytest.fixture(scope="module", params=ARMS, ids=[arm.name for arm in ARMS])
def step(models: dict[str, Model], request: pytest.FixtureRequest) -> Step:
    """Builds one arm's shared page pool and queries."""
    return _step(request.param)


def test_a_mixed_batch_selects_what_its_rows_select_alone(
    step: Step, models: dict[str, Model]
) -> None:
    """Each row of a split batch selects the sets the row selects alone."""
    model = models[step.arm.name]
    alone = {}
    for row in POOL_ORDER:
        record = _record([row])
        assert record[_HEAD_ROWS_SLOT] == 0, (
            f"a one-row batch of {row} must not split"
        )
        alone[row] = _run(model, step, [row], record)
    for order in (POOL_ORDER, POOL_ORDER[::-1]):
        record = _record(order)
        assert record[_HEAD_ROWS_SLOT] > 0, f"{order} does not split"
        got = _run(model, step, order, record)
        for row in order:
            _assert_row_matches(
                got,
                alone[row],
                order,
                row,
                f"{step.arm.name}: the {row} row of {order}",
            )


def test_the_refused_records_keep_the_unsplit_launch(
    step: Step, models: dict[str, Model]
) -> None:
    """A zeros record and an inconsistent one both keep the whole-batch run."""
    model = models[step.arm.name]
    zeros = _run(
        model, step, POOL_ORDER, np.zeros(MIXED_STEP_SPLIT_SIZE, np.uint32)
    )
    refused = _run(model, step, POOL_ORDER, _refused_record(POOL_ORDER))
    why = f"{step.arm.name}: the refused record took another route"
    _assert_same_run(zeros, refused, why)
