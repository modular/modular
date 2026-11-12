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
"""A mixed batch through the dense MLA combined op equals its parts alone.

One decode row (a cached history plus one new token) and one prefill row
(``PREFILL_LEN`` fresh tokens) run through the dense combined op in both row
orders, for the BF16 and the FP8 arm, and each row's output must match the
same row run alone: on the split route the op plans the prefill view's rows
itself, so that view's launch equals the standalone launch. Each row must also
match the same row with the split record zeroed, which sends the decode row
through the prefill arm, and no decode row may be empty. Every record the
split rules refuse keeps the single whole-batch launch, so the refused
outputs agree with each other.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt
import pytest
import torch
from _mla_sparse_graphs import (
    BUFFER_SIZE,
    HIDDEN_SIZE,
    KV_LORA_RANK,
    MAX_BATCH_SIZE,
    PREFILL_LEN,
    Q_LORA_RANK,
    QK_NOPE_HEAD_DIM,
    QK_ROPE_HEAD_DIM,
    ROPE_MAX_SEQ_LEN,
    TOTAL_NUM_PAGES,
    V_HEAD_DIM,
    make_kv_params,
    make_quant_config,
)
from _mla_sparse_test_utils import make_deepseek_rope, random_weights
from max import tree
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.graph.weights import WeightData
from max.nn.kv_cache import KVCacheInputsPerDevice, KVCacheParams
from max.nn.legacy.attention.multi_latent_attention import (
    LatentAttentionWithRope,
)
from max.nn.legacy.attention.multi_latent_attention_fp8 import (
    LatentAttentionWithRopeFp8,
)
from max.pipelines.kv_cache import PagedKVCacheManager
from test_common.context_utils import create_text_context
from torch.utils.dlpack import from_dlpack

NUM_HEADS = 8

Arm = Literal["bf16", "fp8"]
"""A dense weight route of the combined op."""

Row = Literal["decode", "prefill"]
"""A row kind of the mixed batch."""

RowLayout = tuple[Row, ...]
"""The rows of one batch, in the order the batch holds them."""

# Tokens each row kind adds; a decode row reads its cache instead.
_ROW_TOKENS: dict[Row, int] = {"decode": 1, "prefill": PREFILL_LEN}

# Split record slots, as ``read_mixed_split`` reads them.
_HEAD_ROWS, _HEAD_TOKENS, _HEAD_MAX_QUERY = 0, 1, 2

# The host scalar's decode-view split-K count, as ``_mla_mixed_split`` reads it.
_DECODE_PARTITIONS = 1

# The same rows can round differently between launches on bf16, and a split
# decode row takes the decode arm where an unsplit one takes the prefill arm.
ATOL_SAME_ROUTE: dict[Arm, float] = {"bf16": 0.5, "fp8": 0.0}
ATOL_CROSS_ROUTE: dict[Arm, float] = {"bf16": 0.5, "fp8": 2e-4}

# The two arms accumulate the same sum in another order, and bf16's gap passes
# any useful bar at this norm gain, so bf16 compares prefill rows across routes.
CROSS_ROUTE_ROWS: dict[Arm, tuple[Row, ...]] = {
    "bf16": ("prefill",),
    "fp8": ("decode", "prefill"),
}

# MLA RMS-normalizes both latents, so the norm gains alone set the logit
# spread, and flat attention hides a wrong decode-row q.
_NORM_GAIN = 3e3

Leaf = KVCacheInputsPerDevice[Buffer, Buffer]
RecordEdit = Callable[[Leaf], Leaf]


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


def _row_slice(rows: RowLayout, row: Row) -> slice:
    """Returns the output rows of ``row`` in a batch laid out as ``rows``."""
    start = sum(_ROW_TOKENS[r] for r in rows[: rows.index(row)])
    return slice(start, start + _ROW_TOKENS[row])


def _assert_same_values(got: np.ndarray, want: np.ndarray, why: str) -> None:
    """Pins that two runs of the same rows agree bit for bit."""
    np.testing.assert_array_equal(got, want, err_msg=why)


def _assert_row_matches(
    out: np.ndarray,
    rows: RowLayout,
    row: Row,
    alone: np.ndarray,
    atol: float,
    why: str,
) -> None:
    """Pins that ``row`` in a batch equals the same row run alone."""
    np.testing.assert_allclose(
        out[_row_slice(rows, row)], alone, atol=atol, rtol=0.0, err_msg=why
    )


def _assert_decode_row_live(out: np.ndarray, rows: RowLayout, why: str) -> None:
    """Pins that the decode row carries finite, nonzero values."""
    dec = out[_row_slice(rows, "decode")]
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


def _decode_partitions_zero(leaf: Leaf) -> Leaf:
    """Leaves the decode view's split-K count at zero, which is refused."""
    assert leaf.mla_num_partitions is not None
    partitions = leaf.mla_num_partitions.to_numpy().copy()
    partitions[_DECODE_PARTITIONS] = 0
    return dataclasses.replace(
        leaf, mla_num_partitions=Buffer.from_numpy(partitions)
    )


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


def _dense_layer(
    arm: Arm,
) -> tuple[LatentAttentionWithRope | LatentAttentionWithRopeFp8, KVCacheParams]:
    """Builds the dense MLA layer for ``arm`` and its single-cache KV params."""
    mla_params, _ = make_kv_params(NUM_HEADS)
    rope = make_deepseek_rope(NUM_HEADS, ROPE_MAX_SEQ_LEN)
    if arm == "fp8":
        attn: LatentAttentionWithRope | LatentAttentionWithRopeFp8 = (
            LatentAttentionWithRopeFp8(
                rope=rope,
                num_attention_heads=NUM_HEADS,
                num_key_value_heads=1,
                hidden_size=HIDDEN_SIZE,
                kv_params=mla_params,
                quant_config=make_quant_config(),
                devices=[DeviceRef.GPU()],
                graph_mode="auto",
                q_lora_rank=Q_LORA_RANK,
                kv_lora_rank=KV_LORA_RANK,
                qk_nope_head_dim=QK_NOPE_HEAD_DIM,
                qk_rope_head_dim=QK_ROPE_HEAD_DIM,
                v_head_dim=V_HEAD_DIM,
                buffer_size=BUFFER_SIZE,
            )
        )
    else:
        attn = LatentAttentionWithRope(
            rope=rope,
            num_attention_heads=NUM_HEADS,
            num_key_value_heads=1,
            hidden_size=HIDDEN_SIZE,
            kv_params=mla_params,
            dtype=DType.bfloat16,
            devices=[DeviceRef.GPU()],
            graph_mode="auto",
            q_lora_rank=Q_LORA_RANK,
            kv_lora_rank=KV_LORA_RANK,
            qk_nope_head_dim=QK_NOPE_HEAD_DIM,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            v_head_dim=V_HEAD_DIM,
            buffer_size=BUFFER_SIZE,
        )
    return attn, mla_params


@pytest.mark.parametrize("arm", ["bf16", "fp8"])
def test_mixed_batch_splits_into_its_parts(arm: Arm) -> None:
    """A mixed batch through the dense combined op equals its parts alone.

    One decode row (a cached history plus one new token) and one prefill row
    (``PREFILL_LEN`` fresh tokens) run through the combined op in both row
    orders. The record the KV manager fills must give each row the output the
    same row produces alone, because the op plans the prefill view's own
    rows. A zeros record, a zero decode split-K count, and a record whose
    head tokens exceed its rows all keep the single whole-batch launch, so
    the three outputs agree.
    """
    device = Accelerator(0)
    session = InferenceSession(devices=[device])
    attn, kv_params = _dense_layer(arm)
    _ = attn.state_dict()

    # One seeded stream, so every run's rows carry the same values.
    generator = torch.Generator().manual_seed(0)

    def hidden(rows: int) -> torch.Tensor:
        return (
            torch.randn((rows, HIDDEN_SIZE), generator=generator) * 0.02
        ).to(torch.bfloat16)

    hist_hidden = hidden(PREFILL_LEN)
    row_hidden = {"decode": hidden(1), "prefill": hidden(PREFILL_LEN)}

    def to_device(t: torch.Tensor) -> Buffer:
        return Buffer.from_dlpack(t).to(device)

    hidden_type = TensorType(
        DType.bfloat16, ["total_seq_len", HIDDEN_SIZE], DeviceRef.GPU()
    )
    row_offsets_type = TensorType(
        DType.uint32, ["row_offsets_len"], DeviceRef.GPU()
    )
    with Graph(
        f"dense_{arm}_mla_mixed_split",
        input_types=[
            hidden_type,
            row_offsets_type,
            *kv_params.flattened_kv_inputs(),
        ],
    ) as graph:
        hidden_in = graph.inputs[0].tensor
        row_offsets_in = graph.inputs[1].tensor
        kv_collection = kv_params.unflatten_kv_inputs(iter(graph.inputs[2:]))[0]
        out = attn(
            ops.constant(0, DType.uint32, device=DeviceRef.CPU()),
            hidden_in,
            kv_collection,
            freqs_cis=attn.rope.freqs_cis,
            input_row_offsets=row_offsets_in,
        )
        graph.output(out)
    # Fixed weights keep the cross-arm bf16 gap the same on every run.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        weights = random_weights(attn)
    model = session.load(graph, weights_registry=_scaled_norm_gains(weights))

    def run(
        rows: RowLayout, record_edit: RecordEdit | None = None
    ) -> tuple[np.ndarray, npt.NDArray[np.uint32], npt.NDArray[np.int64]]:
        """Rebuilds the decode row's history, then runs the step under test.

        Returns the output values, the split record the step used, and the
        leaf's host scalar.
        """
        kv_manager = PagedKVCacheManager(
            params=kv_params,
            total_num_pages=TOTAL_NUM_PAGES,
            session=session,
            max_batch_size=MAX_BATCH_SIZE,
        )
        decode_ctx = create_text_context(np.empty(PREFILL_LEN))
        prefill_ctx = create_text_context(np.empty(PREFILL_LEN))
        by_row = {"decode": decode_ctx, "prefill": prefill_ctx}
        batch = [by_row[r] for r in rows]
        for ctx in batch:
            kv_manager.claim(ctx)
        if "decode" in rows:
            # The history prefill, then one update per cached token, on a
            # fresh manager each run, so every run's decode cache matches.
            kv_manager.alloc(decode_ctx)
            hist_ri = kv_manager.runtime_inputs([[decode_ctx]])
            model.execute(
                to_device(hist_hidden),
                _offsets_buffer([PREFILL_LEN], device),
                *tree.leaves(hist_ri),
            )
            for _ in range(PREFILL_LEN):
                decode_ctx.update(42)
            kv_manager.step(decode_ctx)
        for ctx in batch:
            kv_manager.alloc(ctx)
        leaf = kv_manager.runtime_inputs_for_leaf([batch])[0]
        if record_edit is not None:
            leaf = record_edit(leaf)
        assert leaf.mixed_step_split is not None
        assert leaf.mla_num_partitions is not None
        out = model.execute(
            to_device(torch.cat([row_hidden[r] for r in rows])),
            _offsets_buffer([_ROW_TOKENS[r] for r in rows], device),
            *tree.leaves(leaf),
        )[0]
        return (
            _output_values(out),
            leaf.mixed_step_split.to_numpy(),
            leaf.mla_num_partitions.to_numpy(),
        )

    dec_first: RowLayout = ("decode", "prefill")
    pref_first: RowLayout = ("prefill", "decode")

    # Runs 1 and 2: the mixed batch, both row orders, the manager's record.
    out_dec_first, record, partitions = run(dec_first)
    # Slot 0 names the head view's rows and element 1 of the host scalar the
    # decode view's partition count; a zeros pair would prove nothing.
    assert record[_HEAD_ROWS] > 0, "the manager's record does not split"
    assert partitions[_DECODE_PARTITIONS] > 0, "the host scalar's count is zero"

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
        ATOL_SAME_ROUTE[arm],
        "the split decode row differs from the same row run alone",
    )
    _assert_row_matches(
        out_dec_first,
        dec_first,
        "prefill",
        pref_alone,
        ATOL_SAME_ROUTE[arm],
        "the split prefill row differs from the same row run alone",
    )
    _assert_row_matches(
        out_pref_first,
        pref_first,
        "prefill",
        pref_alone,
        ATOL_SAME_ROUTE[arm],
        "the reordered split's prefill row differs from the row alone",
    )
    _assert_row_matches(
        out_pref_first,
        pref_first,
        "decode",
        dec_alone,
        ATOL_SAME_ROUTE[arm],
        "the reordered split's decode row differs from the row alone",
    )

    # Run 4: the refused edits keep the single whole-batch launch, so the
    # three outputs agree with each other.
    refused = [
        run(dec_first, edit)[0]
        for edit in (
            _without_split,
            _decode_partitions_zero,
            _head_tokens_past_rows,
        )
    ]
    _assert_same_values(
        refused[0],
        refused[1],
        "the zeros record and the zero split-K count took different routes",
    )
    _assert_same_values(
        refused[0],
        refused[2],
        "the record past its rows took a different route",
    )

    # Run 5: the zeros record is the unsplit route, which sends the decode row
    # through the prefill arm, so every split row compared across routes must
    # equal it too.
    unsplit = refused[0]
    for row in CROSS_ROUTE_ROWS[arm]:
        np.testing.assert_allclose(
            out_dec_first[_row_slice(dec_first, row)],
            unsplit[_row_slice(dec_first, row)],
            atol=ATOL_CROSS_ROUTE[arm],
            rtol=0.0,
            err_msg=(
                f"the split {row} row differs from the same row without a split"
            ),
        )

    # A decode row that never ran would compare equal everywhere.
    dec_alone_rows: RowLayout = ("decode",)
    for why, values, rows in (
        ("the split batch", out_dec_first, dec_first),
        ("the reordered split batch", out_pref_first, pref_first),
        ("the decode row alone", dec_alone, dec_alone_rows),
        ("the unsplit batch", unsplit, dec_first),
    ):
        _assert_decode_row_live(values, rows, why)
