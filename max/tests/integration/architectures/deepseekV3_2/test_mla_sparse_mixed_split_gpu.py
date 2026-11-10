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
"""A mixed batch through the sparse MLA combined op equals its parts alone.

The graphs are the CPU-precompiled MEFs of ``:mla_sparse_prefill_mefs``; this
test does not compile. One decode row (a cached history plus one new token) and
one prefill row (``PREFILL_LEN`` fresh tokens) run through the combined op in
both row orders, and each row's output must match the same row run alone. Each
row must also match the same row with the split record zeroed, which sends the
decode row through the prefill arm, and no decode row may be empty. Every
record the split rules refuse keeps the single whole-batch launch, so the three
refused outputs agree with each other. A tail view of two prefill rows, and the
mirror of two decode rows, run the same per-row compares, which is where the
rebased offsets matter. The NVFP4 param runs the same rows over a packed latent
cache, whose prefill view converts the context to FP8 once. A serving verify
step runs beside a prefill row too, with decode rows of ``1 + K`` tokens over a
cached history long enough to split the decode view.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt
import pytest
import torch
from _mla_sparse_graphs import (
    HIDDEN_SIZE,
    MAX_BATCH_SIZE,
    PAGE_SIZE,
    PREFILL_LEN,
    SPECS_BY_NAME,
    TOTAL_NUM_PAGES,
    PrefillSpec,
    make_multi_kv,
    weights_for,
)
from max import tree
from max.driver import Accelerator, Buffer
from max.engine import InferenceSession, Model
from max.graph.weights import WeightData
from max.nn.kv_cache import KVCacheInputsPerDevice, MultiKVCacheParams
from max.pipelines.kv_cache import PagedKVCacheManager
from test_common.context_utils import create_text_context
from test_common.mef_precompile import init_from_mef, mefs_from_env
from torch.utils.dlpack import from_dlpack

# The indexer's top-k order varies between launches, so equal inputs can pick
# other keys: bf16 and nvfp4 need slack on both compares, fp8 only across
# routes (nvfp4's decode row moves by a few output ulps).
ATOL_SAME_ROUTE: dict[str, float] = {
    "fp8": 0.0,
    "bf16": 3.125e-1,
    "nvfp4": 4.0,
}
ATOL_CROSS_ROUTE: dict[str, float] = {
    "fp8": 1e-4,
    "bf16": 3.125e-1,
    "nvfp4": 4.0,
}

# MLA RMS-normalizes both latents, so the norm gains alone set the logit
# spread, and flat attention hides a wrong decode-row q.
_NORM_GAIN = 3e3

Row = Literal["decode", "prefill"]
"""A row kind of the mixed batch."""

RowLayout = tuple[Row, ...]
"""The rows of one batch, in the order the batch holds them."""

# Tokens each row kind adds; a decode row reads its cache instead.
_ROW_TOKENS: dict[Row, int] = {"decode": 1, "prefill": PREFILL_LEN}

# Every row kind, in a fixed order, so a run's seeded values do not move with
# the batch layout.
_ROW_KINDS: tuple[Row, ...] = ("decode", "prefill")

# Drafts a serving verify step puts in front of its new token.
_VERIFY_DRAFTS = 5

_VERIFY_TOKENS: dict[Row, int] = {
    "decode": 1 + _VERIFY_DRAFTS,
    "prefill": PREFILL_LEN,
}

# A decode history the split-K heuristic lifts past one partition: its floor is
# four split pages, and a page holds PAGE_SIZE cached tokens.
_SPLIT_K_HISTORY = 8 * PAGE_SIZE

# A prefill row that the view's one-wave bucket splits in two: the split length
# is the view's cache bound plus this width, which must pass one split page,
# and two partitions must still fit one wave.
_WIDE_PREFILL_TOKENS = 74

# A cached history of two pages for the decode row beside it.
_PREFILL_SPLIT_HISTORY = 2 * PAGE_SIZE

# The split case's row widths: a wide prefill head view and a verify tail.
_PREFILL_SPLIT_WIDTHS: dict[Row, int] = {
    "prefill": _WIDE_PREFILL_TOKENS,
    "decode": 1 + _VERIFY_DRAFTS,
}

# Workspace byte sizes of the head view at two partitions, in float32 elements:
# its bf16 partial outputs, then its float32 LSE, plus blocks that bracket them.
# The driver hands a freed allocation back only for its own size.
_POISON_SIZES: tuple[int, ...] = (
    8 * 303104,
    606208,
    303104,
    1184,
    1024,
)

# Poisoned step runs, so a lucky workspace contents cannot pass the case.
_POISON_REPEATS = 3

# A tail view of two rows: a one-row view addresses the same tokens with or
# without the rebase, so only a wider tail proves it.
_TWO_ROW_TAIL_LAYOUTS: tuple[RowLayout, ...] = (
    ("decode", "prefill", "prefill"),
    ("prefill", "decode", "decode"),
)

# Split record slots, as ``read_mixed_split`` reads them.
_HEAD_ROWS, _HEAD_TOKENS, _HEAD_MAX_QUERY = 0, 1, 2

Leaf = KVCacheInputsPerDevice[Buffer, Buffer]
RecordEdit = Callable[[Leaf], Leaf]


def _fixed_weights_for(spec: PrefillSpec) -> dict[str, WeightData]:
    """Returns ``spec``'s weights from a fixed seed, so every run compares
    one instance."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        return weights_for(spec)


def _output_values(out: Buffer) -> np.ndarray:
    """Returns an output buffer as float values for tolerance compares."""
    out_t = from_dlpack(out).cpu()
    return (
        out_t.float().numpy()
        if out_t.dtype == torch.bfloat16
        else out_t.numpy()
    )


def _offsets_buffer(lengths: Sequence[int], device: Accelerator) -> Buffer:
    """Returns one batch's row offsets, device-side."""
    lengths_np = np.array(lengths, dtype=np.uint32)
    offsets = np.concatenate(
        (np.zeros(1, dtype=np.uint32), np.cumsum(lengths_np, dtype=np.uint32))
    )
    return Buffer.from_numpy(offsets).to(device)


def _batch_context_length(positions: Sequence[int]) -> Buffer:
    """Returns the host batch context length for one batch's positions.

    Page-aligned and summed, as the model's batch processor computes it.
    """
    length = sum(
        (position + PAGE_SIZE - 1) // PAGE_SIZE * PAGE_SIZE
        for position in positions
    )
    return Buffer.from_numpy(np.array([length], dtype=np.int32))


def _poison_recycled_memory(device: Accelerator) -> None:
    """NaNs the device memory the step's split-K workspaces are about to take.

    The driver hands a freed block back only after the queue drains its free,
    so each size is freed and the queue synced. Sizes bracket both workspaces:
    a slot the kernels never write then holds NaN, which no value compare
    passes.
    """
    for floats in _POISON_SIZES:
        nan = Buffer.from_numpy(np.full(floats, np.nan, dtype=np.float32)).to(
            device
        )
        del nan
        torch.cuda.synchronize()


def _row_slices(
    rows: RowLayout, row: Row, tokens: Mapping[Row, int] = _ROW_TOKENS
) -> list[slice]:
    """Returns the output rows of every ``row`` in a batch laid out as
    ``rows``."""
    starts = [0]
    for kind in rows:
        starts.append(starts[-1] + tokens[kind])
    return [
        slice(starts[i], starts[i + 1])
        for i, kind in enumerate(rows)
        if kind == row
    ]


def _row_slice(
    rows: RowLayout, row: Row, tokens: Mapping[Row, int] = _ROW_TOKENS
) -> slice:
    """Returns the output rows of the first ``row`` in a batch laid out as
    ``rows``."""
    return _row_slices(rows, row, tokens)[0]


def _assert_same_values(
    got: np.ndarray, want: np.ndarray, atol: float, why: str
) -> None:
    """Pins that two runs of the same rows agree within ``atol``.

    ``atol`` 0 pins exact equality.
    """
    np.testing.assert_allclose(got, want, atol=atol, rtol=0.0, err_msg=why)


def _assert_row_matches(
    out: np.ndarray,
    rows: RowLayout,
    row: Row,
    alone: np.ndarray,
    atol: float,
    why: str,
    tokens: Mapping[Row, int] = _ROW_TOKENS,
) -> None:
    """Pins that every ``row`` in a batch equals the same row run alone."""
    slices = _row_slices(rows, row, tokens)
    assert slices, f"the batch has no {row} row"
    for rows_slice in slices:
        _assert_same_values(out[rows_slice], alone, atol, why)


def _assert_decode_row_live(
    out: np.ndarray,
    rows: RowLayout,
    why: str,
    tokens: Mapping[Row, int] = _ROW_TOKENS,
) -> None:
    """Pins that the decode rows carry finite, nonzero values."""
    for rows_slice in _row_slices(rows, "decode", tokens):
        dec = out[rows_slice]
        assert np.isfinite(dec).all(), f"{why}: the decode row is not finite"
        assert np.any(dec != 0), f"{why}: the decode row is all zero"


def _record_copy(leaf: Leaf) -> npt.NDArray[np.uint32]:
    """Returns a writable copy of the leaf's split record."""
    assert leaf.mixed_step_split is not None
    return leaf.mixed_step_split.to_numpy().copy()


def _with_record(leaf: Leaf, record: npt.NDArray[np.uint32]) -> Leaf:
    """Returns the leaf with ``record`` in place of its split record."""
    return dataclasses.replace(leaf, mixed_step_split=Buffer.from_numpy(record))


def _without_split(leaf: Leaf) -> Leaf:
    """Drops the head view, so the step keeps the single whole-batch launch."""
    return _with_record(leaf, np.zeros_like(_record_copy(leaf)))


def _head_tokens_below_rows(leaf: Leaf) -> Leaf:
    """Leaves the head view fewer tokens than it has rows, which is refused."""
    record = _record_copy(leaf)
    record[_HEAD_TOKENS] = 0
    return _with_record(leaf, record)


def _head_tokens_past_rows(leaf: Leaf) -> Leaf:
    """Gives the head view more tokens than its rows hold, which is refused."""
    record = _record_copy(leaf)
    # The offsets give each head row at most the view's max query length.
    record[_HEAD_TOKENS] = record[_HEAD_ROWS] * record[_HEAD_MAX_QUERY] + 1
    return _with_record(leaf, record)


def _scaled_norm_gains(
    weights: dict[str, WeightData],
) -> dict[str, WeightData]:
    """Multiplies every layernorm gain by ``_NORM_GAIN``."""
    scaled: dict[str, WeightData] = {}
    for name, weight in weights.items():
        if not name.endswith("layernorm.weight"):
            scaled[name] = weight
            continue
        values = from_dlpack(weight.data)
        scaled_values = (values.float() * _NORM_GAIN).to(values.dtype)
        scaled[name] = WeightData(
            Buffer.from_dlpack(scaled_values),
            name,
            weight.dtype,
            weight.shape,
        )
    return scaled


def _run_staged_step(
    model: Model,
    session: InferenceSession,
    multi_kv: MultiKVCacheParams,
    device: Accelerator,
    rows: RowLayout,
    tokens: Mapping[Row, int],
    history: int,
    *,
    record_edit: RecordEdit | None = None,
    poison_workspaces: bool = False,
) -> tuple[np.ndarray, npt.NDArray[np.uint32], npt.NDArray[np.int64]]:
    """Rebuilds each decode row's history, then runs the step under test.

    ``history`` is the cached-token count each decode row reaches before the
    step. ``poison_workspaces`` NaNs the memory the step's split-K workspaces
    are about to take.

    Returns the output values, the split record the step used, and the
    leaf's host scalar.
    """
    kv_manager = PagedKVCacheManager(
        params=multi_kv,
        total_num_pages=TOTAL_NUM_PAGES,
        session=session,
        max_batch_size=MAX_BATCH_SIZE,
    )
    # One context per row: a repeated row kind needs its own cache line,
    # holding the same tokens. A decode row holds its whole history, a prefill
    # row its query.
    batch = [
        create_text_context(
            np.empty(history if row == "decode" else tokens[row]),
            max_length=history + tokens[row],
        )
        for row in rows
    ]
    for ctx in batch:
        kv_manager.claim(ctx)

    # One seeded stream, so every run's rows carry the same values, drawn in a
    # fixed kind order so a run's values do not move with the layout.
    generator = torch.Generator().manual_seed(0)

    def to_device(t: torch.Tensor) -> Buffer:
        return Buffer.from_dlpack(t).to(device)

    def hidden(count: int) -> torch.Tensor:
        values = torch.randn((count, HIDDEN_SIZE), generator=generator) * 0.02
        return values.to(torch.bfloat16)

    hist_hidden = to_device(hidden(history))
    query_hidden = {kind: hidden(tokens[kind]) for kind in _ROW_KINDS}

    for ctx, row in zip(batch, rows, strict=True):
        if row != "decode":
            continue
        # The history prefill, then the one update that takes the processing
        # window past it, on a fresh manager each run, so every run's decode
        # cache matches.
        kv_manager.alloc(ctx)
        hist_ri = kv_manager.runtime_inputs([[ctx]])
        assert isinstance(hist_ri, dict)
        model.execute(
            hist_hidden,
            _offsets_buffer([history], device),
            *tree.leaves(hist_ri),
            _batch_context_length([ctx.tokens.current_position]),
        )
        ctx.update(42)
        kv_manager.step(ctx)
        # The update leaves one token active; a wider row verifies the rest of
        # its width as drafts behind it.
        ctx.tokens.rewind_processing(tokens[row] - 1)

    for ctx in batch:
        kv_manager.alloc(ctx)
    kv_ri = kv_manager.runtime_inputs([batch])
    assert isinstance(kv_ri, dict)
    mla = kv_ri["mla"]
    assert isinstance(mla, tuple) and len(mla) == 1
    leaf = mla[0]
    assert isinstance(leaf, KVCacheInputsPerDevice)
    if record_edit is not None:
        # In place, so the leaves keep the order the graph declares.
        leaf = record_edit(leaf)
        kv_ri["mla"] = (leaf,)
    assert leaf.mixed_step_split is not None
    assert leaf.mla_num_partitions is not None
    if poison_workspaces:
        _poison_recycled_memory(device)
    out = model.execute(
        to_device(torch.cat([query_hidden[r] for r in rows])),
        _offsets_buffer([tokens[r] for r in rows], device),
        *tree.leaves(kv_ri),
        _batch_context_length([ctx.tokens.current_position for ctx in batch]),
    )[0]
    return (
        _output_values(out),
        leaf.mixed_step_split.to_numpy(),
        leaf.mla_num_partitions.to_numpy(),
    )


@pytest.mark.parametrize(
    "spec_name", ["prefill_bf16_8h", "prefill_fp8_8h", "prefill_nvfp4_8h"]
)
def test_mixed_batch_splits_into_its_parts(spec_name: str) -> None:
    """A mixed batch through the sparse combined op equals its parts run alone.

    One decode row (a cached history plus one new token) and one prefill row
    (``PREFILL_LEN`` fresh tokens) run through the precompiled combined op in
    both row orders. The record the KV manager fills must give each row the
    output the same row produces alone: the sparse prefill arm reads no plan,
    so a view's launch equals the standalone launch. A zeros record, a record
    without head tokens, and a record whose head tokens exceed its rows all
    keep the single whole-batch launch, so the three outputs agree. A tail
    view of two rows, in both directions, runs the same per-row compares.
    """
    spec = SPECS_BY_NAME[spec_name]
    same_route_atol = ATOL_SAME_ROUTE[spec.route]
    cross_route_atol = ATOL_CROSS_ROUTE[spec.route]
    mef_path = mefs_from_env("MLA_MEF_RLOCATIONS")[f"{spec.name}.mef"]
    assert mef_path.is_file(), f"precompiled MEF missing: {mef_path}"

    device = Accelerator(0)
    session = InferenceSession(devices=[Accelerator()])
    model = init_from_mef(
        session, mef_path, _scaled_norm_gains(_fixed_weights_for(spec))
    )
    multi_kv = make_multi_kv(spec.num_heads, spec.cache_dtype)

    def run(
        rows: RowLayout, record_edit: RecordEdit | None = None
    ) -> tuple[np.ndarray, npt.NDArray[np.uint32], npt.NDArray[np.int64]]:
        return _run_staged_step(
            model,
            session,
            multi_kv,
            device,
            rows,
            _ROW_TOKENS,
            PREFILL_LEN,
            record_edit=record_edit,
        )

    dec_first: RowLayout = ("decode", "prefill")
    pref_first: RowLayout = ("prefill", "decode")

    # Runs 1 and 2: the mixed batch, both row orders, the manager's record.
    out_dec_first, record, partitions = run(dec_first)
    # Slot 0 names the head view's rows and element 1 of the host scalar the
    # decode view's partition count; a zeros pair would prove nothing.
    assert record[_HEAD_ROWS] > 0, "the manager's record does not split"
    assert partitions[1] > 0, "the host scalar's decode count is zero"

    out_pref_first, pref_first_record, _ = run(pref_first)
    assert pref_first_record[_HEAD_ROWS] > 0, (
        "the reordered record does not split"
    )

    # Run 3: each row alone, a batch of one kind keeps the zeros record.
    dec_alone, dec_record, _ = run(("decode",))
    assert dec_record[_HEAD_ROWS] == 0, "a decode-only step must not split"
    pref_alone, pref_record, _ = run(("prefill",))
    assert pref_record[_HEAD_ROWS] == 0, "a prefill-only step must not split"

    # Runs 1 and 2 equal run 3, row by row.
    _assert_row_matches(
        out_dec_first,
        dec_first,
        "decode",
        dec_alone,
        same_route_atol,
        "the split decode row differs from the same row run alone",
    )
    _assert_row_matches(
        out_dec_first,
        dec_first,
        "prefill",
        pref_alone,
        same_route_atol,
        "the split prefill row differs from the same row run alone",
    )
    _assert_row_matches(
        out_pref_first,
        pref_first,
        "prefill",
        pref_alone,
        same_route_atol,
        "the reordered split's prefill row differs from the row alone",
    )
    _assert_row_matches(
        out_pref_first,
        pref_first,
        "decode",
        dec_alone,
        same_route_atol,
        "the reordered split's decode row differs from the row alone",
    )

    # Runs 6 and 7: a tail view of two rows, both directions; with one row the
    # whole-batch offsets address the same tokens as the rebased ones.
    tail_batches: list[tuple[str, np.ndarray, RowLayout]] = []
    for layout in _TWO_ROW_TAIL_LAYOUTS:
        out, tail_record, _ = run(layout)
        assert tail_record[_HEAD_ROWS] == len(layout) - 2, (
            "a three-row batch did not split into a two-row tail"
        )
        _assert_row_matches(
            out,
            layout,
            "decode",
            dec_alone,
            same_route_atol,
            "the split decode row of a two-row tail differs from the row alone",
        )
        _assert_row_matches(
            out,
            layout,
            "prefill",
            pref_alone,
            same_route_atol,
            "the split prefill row of a two-row tail differs from the row alone",
        )
        tail_batches.append((f"the two-row-tail batch {layout}", out, layout))

    # Run 4: the refused edits keep the single whole-batch launch, so the
    # three outputs agree with each other.
    refused = [
        run(dec_first, edit)[0]
        for edit in (
            _without_split,
            _head_tokens_below_rows,
            _head_tokens_past_rows,
        )
    ]
    _assert_same_values(
        refused[0],
        refused[1],
        cross_route_atol,
        "the zeros record and the tokenless record took different routes",
    )
    _assert_same_values(
        refused[0],
        refused[2],
        cross_route_atol,
        "the record past its rows took a different route",
    )

    # Run 5: the zeros record is the unsplit route, which sends the decode row
    # through the prefill arm, so every split row must equal it too.
    unsplit = refused[0]
    for row in dec_first:
        _assert_same_values(
            out_dec_first[_row_slice(dec_first, row)],
            unsplit[_row_slice(dec_first, row)],
            cross_route_atol,
            f"the split {row} row differs from the same row without a split",
        )

    # A decode row that never ran would compare equal everywhere.
    dec_alone_rows: RowLayout = ("decode",)
    for why, values, rows in (
        ("the split batch", out_dec_first, dec_first),
        ("the reordered split batch", out_pref_first, pref_first),
        ("the decode row alone", dec_alone, dec_alone_rows),
        ("the unsplit batch", unsplit, dec_first),
        *tail_batches,
    ):
        _assert_decode_row_live(values, rows, why)


@pytest.mark.parametrize(
    "rows, history",
    [
        pytest.param(("decode", "prefill"), PREFILL_LEN, id="decode-head"),
        pytest.param(("prefill", "decode"), PREFILL_LEN, id="decode-tail"),
        pytest.param(("decode", "prefill"), _SPLIT_K_HISTORY, id="split-k"),
        pytest.param(
            ("decode",) * 7 + ("prefill",), PREFILL_LEN, id="eight-rows"
        ),
    ],
)
def test_mixed_verify_row_splits_into_its_parts(
    rows: RowLayout, history: int
) -> None:
    """A verify-width decode row split off a mixed batch equals its parts alone.

    The rows are a serving verify step's: ``1 + K`` tokens per decode row over a
    cached history, beside one prefill row. Every row must equal the same row run
    alone, with the decode view at either end of the batch and with a history
    deep enough to split the decode view into several partitions.
    """
    spec = SPECS_BY_NAME["prefill_fp8_8h"]
    same_route_atol = ATOL_SAME_ROUTE[spec.route]
    mef_path = mefs_from_env("MLA_MEF_RLOCATIONS")[f"{spec.name}.mef"]
    assert mef_path.is_file(), f"precompiled MEF missing: {mef_path}"

    device = Accelerator(0)
    session = InferenceSession(devices=[Accelerator()])
    model = init_from_mef(
        session, mef_path, _scaled_norm_gains(_fixed_weights_for(spec))
    )
    # The record builder takes its decode width from the draft count, which a
    # verify step widens to the row it verifies.
    multi_kv = dataclasses.replace(
        make_multi_kv(spec.num_heads, spec.cache_dtype),
        num_draft_tokens=_VERIFY_DRAFTS,
    )

    def run(
        rows: RowLayout,
    ) -> tuple[np.ndarray, npt.NDArray[np.uint32], npt.NDArray[np.int64]]:
        return _run_staged_step(
            model, session, multi_kv, device, rows, _VERIFY_TOKENS, history
        )

    out, record, partitions = run(rows)
    # Slot 0 names the head view's rows; a zeros record would prove nothing.
    assert record[_HEAD_ROWS] > 0, "the manager's record does not split"
    # Element 1 of the host scalar is the decode view's partition count, which
    # the split-K floor lifts above one once the cache covers four split pages.
    assert partitions[1] >= (2 if history >= _SPLIT_K_HISTORY else 1), (
        f"the decode view staged {partitions[1]} partitions over a "
        f"{history}-token history"
    )

    dec_alone = run(("decode",))[0]
    pref_alone = run(("prefill",))[0]
    _assert_row_matches(
        out,
        rows,
        "decode",
        dec_alone,
        same_route_atol,
        "the split verify row differs from the same row run alone",
        _VERIFY_TOKENS,
    )
    _assert_row_matches(
        out,
        rows,
        "prefill",
        pref_alone,
        same_route_atol,
        "the split prefill row differs from the same row run alone",
        _VERIFY_TOKENS,
    )
    assert np.isfinite(out).all(), "the split batch has a non-finite row"
    _assert_decode_row_live(out, rows, "the split batch", _VERIFY_TOKENS)


def test_prefill_view_splits_on_its_own_dispatch_triple() -> None:
    """A wide prefill head view split beside a verify tail equals its parts.

    The head view's split-K buffers are sized and combined with its own rows,
    so every value must survive a workspace poison that only a kernel striding
    them by its own rows overwrites. One prefill row of ``_WIDE_PREFILL_TOKENS``
    leads, one verify-width decode row over ``_PREFILL_SPLIT_HISTORY`` cached
    tokens trails, so the head view's one-wave bucket splits it in two.
    """
    spec = SPECS_BY_NAME["prefill_fp8_fp8kv_8h_topk256"]
    same_route_atol = ATOL_SAME_ROUTE[spec.route]
    mef_path = mefs_from_env("MLA_MEF_RLOCATIONS")[f"{spec.name}.mef"]
    assert mef_path.is_file(), f"precompiled MEF missing: {mef_path}"

    device = Accelerator(0)
    session = InferenceSession(devices=[Accelerator()])
    model = init_from_mef(
        session, mef_path, _scaled_norm_gains(_fixed_weights_for(spec))
    )
    multi_kv = dataclasses.replace(
        make_multi_kv(spec.num_heads, spec.cache_dtype),
        num_draft_tokens=_VERIFY_DRAFTS,
    )

    rows: RowLayout = ("prefill", "decode")

    # The bug needs the head view in two partitions or more, which takes a split
    # length past one page.
    split_len = min(
        spec.index_topk, _PREFILL_SPLIT_HISTORY + _WIDE_PREFILL_TOKENS
    )
    assert split_len > PAGE_SIZE, (
        f"the prefill view's split length {split_len} fits one page, so the "
        "view does not split and the case cannot reach the bug"
    )

    def run(
        rows: RowLayout, *, poison: bool = False
    ) -> tuple[np.ndarray, npt.NDArray[np.uint32], npt.NDArray[np.int64]]:
        return _run_staged_step(
            model,
            session,
            multi_kv,
            device,
            rows,
            _PREFILL_SPLIT_WIDTHS,
            _PREFILL_SPLIT_HISTORY,
            poison_workspaces=poison,
        )

    pref_alone = run(("prefill",))[0]
    dec_alone = run(("decode",))[0]

    # Every poisoned run must match, not only the first: a slot the head view
    # never writes stays NaN, so a launch striding it by the batch's rows fails
    # each time.
    for attempt in range(_POISON_REPEATS):
        out, record, _ = run(rows, poison=True)
        why = f"poisoned run {attempt}"
        assert record[_HEAD_ROWS] == 1, "the manager's record does not split"
        assert record[_HEAD_MAX_QUERY] == _WIDE_PREFILL_TOKENS, (
            "the head view is not the prefill view"
        )
        assert np.isfinite(out).all(), f"{why}: the batch is not finite"
        _assert_row_matches(
            out,
            rows,
            "prefill",
            pref_alone,
            same_route_atol,
            f"{why}: the head-view prefill row differs from the row alone",
            _PREFILL_SPLIT_WIDTHS,
        )
        _assert_row_matches(
            out,
            rows,
            "decode",
            dec_alone,
            same_route_atol,
            f"{why}: the tail decode row differs from the row alone",
            _PREFILL_SPLIT_WIDTHS,
        )
