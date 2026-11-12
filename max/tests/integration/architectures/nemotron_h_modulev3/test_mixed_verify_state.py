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
"""A mixed verify commits each row by the row rule.

The verify arm runs one Mamba layer's conv and scan the way a verify step
does, then the replay fold. A row longer than ``K + 1`` tokens is a prefill
row: the verify commits its state at its last real token and the fold
skips it; decode rows keep their decode path.
"""

from __future__ import annotations

import max.driver as md
import numpy as np
import pytest
import torch
from max.driver import Accelerator, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    Graph,
    TensorType,
    TensorValue,
    ops,
)
from max.nn._graph_ops.state_space import (
    causal_conv1d_varlen_fwd,
    causal_conv1d_varlen_fwd_rows,
    mamba2_replay_fold,
    mamba2_ssd_chunk_scan_varlen_fwd_inplace,
    mamba2_ssd_chunk_scan_varlen_fwd_inplace_rows,
    verify_width_operand,
)
from max.pipelines.architectures.nemotron_h_modulev3.state_rollback import (
    replay_plan,
    verify_update_state,
    verify_window,
)
from max.pipelines.speculative.config import MAGIC_DRAFT_TOKEN_ID

_NHEADS = 8
_HEAD_DIM = 64
_NGROUPS = 2
_DSTATE = 128
_WIDTH = 4
_INTERMEDIATE = _NHEADS * _HEAD_DIM
_CONV_DIM = _INTERMEDIATE + 2 * _NGROUPS * _DSTATE
_PROJECTED = _INTERMEDIATE + _CONV_DIM + _NHEADS
_SLOTS = 12
_SPARE = _SLOTS - 1
"""A pool row no request owns, for a ragged launch with no request."""


def _scan_inputs(
    convolved: TensorValue, projected: TensorValue
) -> tuple[TensorValue, TensorValue, TensorValue, TensorValue]:
    """The scan's ``x``, ``B``, ``C`` and ``dt`` as column slices."""
    group_dim = _NGROUPS * _DSTATE
    x, B, C = ops.split(
        convolved, [_INTERMEDIATE, group_dim, group_dim], axis=1
    )
    return (
        ops.reshape(x, [-1, _NHEADS, _HEAD_DIM]),
        ops.reshape(B, [-1, _NGROUPS, _DSTATE]),
        ops.reshape(C, [-1, _NGROUPS, _DSTATE]),
        projected[:, _INTERMEDIATE + _CONV_DIM :],
    )


def _layer_rows(
    projected: TensorValue,
    conv_weight: TensorValue,
    conv_bias: TensorValue,
    A: TensorValue,
    dt_bias: TensorValue,
    D: TensorValue,
    conv_pool: BufferValue,
    ssm_pool: BufferValue,
    offsets: TensorValue,
    conv_rows: TensorValue,
    ssm_rows: TensorValue,
    has_initial_state: TensorValue,
    update_state: TensorValue,
    verify_width: TensorValue,
) -> tuple[TensorValue, TensorValue]:
    """One Mamba layer's two state updates, prefill rows committing in place."""
    convolved = causal_conv1d_varlen_fwd_rows(
        projected[:, _INTERMEDIATE : _INTERMEDIATE + _CONV_DIM],
        conv_weight,
        conv_bias,
        conv_pool,
        offsets,
        conv_rows,
        has_initial_state,
        update_state,
        verify_width,
        channels_last=True,
    )
    x, B, C, dt = _scan_inputs(convolved, projected)
    y = mamba2_ssd_chunk_scan_varlen_fwd_inplace_rows(
        x=x,
        dt=dt,
        A=A,
        B=B,
        C=C,
        D=D,
        dt_bias=dt_bias,
        ssm_pool=ssm_pool,
        query_start_loc=offsets,
        has_initial_state=has_initial_state,
        cache_indices=ssm_rows,
        update_state=update_state,
        xbc=convolved,
        verify_width=verify_width,
    )
    return convolved, y


def _layer_forward(
    projected: TensorValue,
    conv_weight: TensorValue,
    conv_bias: TensorValue,
    A: TensorValue,
    dt_bias: TensorValue,
    D: TensorValue,
    conv_pool: BufferValue,
    ssm_pool: BufferValue,
    offsets: TensorValue,
    conv_rows: TensorValue,
    ssm_rows: TensorValue,
    has_initial_state: TensorValue,
    update_state: TensorValue,
) -> tuple[TensorValue, TensorValue]:
    """One Mamba layer's two state updates, every row updating its slot."""
    convolved = causal_conv1d_varlen_fwd(
        projected[:, _INTERMEDIATE : _INTERMEDIATE + _CONV_DIM],
        conv_weight,
        conv_bias,
        conv_pool,
        offsets,
        conv_rows,
        has_initial_state,
        channels_last=True,
        update_state=update_state,
    )
    x, B, C, dt = _scan_inputs(convolved, projected)
    y = mamba2_ssd_chunk_scan_varlen_fwd_inplace(
        x=x,
        dt=dt,
        A=A,
        B=B,
        C=C,
        D=D,
        dt_bias=dt_bias,
        ssm_pool=ssm_pool,
        query_start_loc=offsets,
        has_initial_state=has_initial_state,
        cache_indices=ssm_rows,
        update_state=update_state,
        xbc=convolved,
    )
    return convolved, y


def _build_graph(device: DeviceRef) -> Graph:
    conv_pool = BufferType(
        DType.bfloat16, [_SLOTS, _CONV_DIM, _WIDTH - 1], device=device
    )
    ssm_pool = BufferType(
        DType.float32, [_SLOTS, _NHEADS, _HEAD_DIM, _DSTATE], device=device
    )
    rows = TensorType(DType.uint32, [1, "batch_size"], device=device)

    def group(name: str) -> list[TensorType]:
        """One ragged launch's rows, their offsets and pool rows."""
        return [
            TensorType(DType.int64, [f"{name}_rows"], device),
            TensorType(DType.int32, [f"{name}_offsets"], device),
            TensorType(DType.uint32, [f"{name}_batch"], device),
            TensorType(DType.uint32, [f"{name}_batch"], device),
        ]

    with Graph(
        "nemotron_h_mixed_verify_state",
        input_types=[
            conv_pool,
            ssm_pool,
            conv_pool,
            ssm_pool,
            conv_pool,
            ssm_pool,
            rows,
            rows,
            TensorType(DType.int64, ["batch_size", "num_steps"], device),
            TensorType(DType.uint32, ["offsets"], device),
            TensorType(DType.uint32, ["batch_size"], device),
            TensorType(DType.bfloat16, ["rows", _PROJECTED], device),
            TensorType(DType.int64, ["kept"], device),
            *group("prefill"),
            *group("decode"),
            *group("decode_only"),
            TensorType(DType.bfloat16, [_NHEADS], device),
            TensorType(DType.bfloat16, [_NHEADS], device),
            TensorType(DType.bfloat16, [_CONV_DIM], device),
            TensorType(DType.bfloat16, [_NHEADS], device),
            TensorType(DType.bfloat16, [_CONV_DIM, _WIDTH], device),
        ],
    ) as graph:
        pools = [v.buffer for v in graph.inputs[:6]]
        (
            conv_rows,
            ssm_rows,
            draft_tokens,
            merged_offsets,
            num_accepted,
            projected,
            kept_rows,
        ) = (v.tensor for v in graph.inputs[6:13])
        groups = [v.tensor for v in graph.inputs[13:25]]
        A, dt_bias, conv_bias, D, conv_weight = (
            v.tensor for v in graph.inputs[25:]
        )
        k = draft_tokens.shape[1]
        batch = conv_rows.shape[1]

        # The verify arm: the row rule commits prefill rows in place.
        verify_width = verify_width_operand(k)
        has_initial_state = ops.constant(
            True, DType.bool, device=device
        ).broadcast_to([batch])
        offsets = merged_offsets.cast(DType.int32)
        convolved, y = _layer_rows(
            projected,
            conv_weight,
            conv_bias,
            A,
            dt_bias,
            D,
            pools[0],
            pools[1],
            offsets,
            conv_rows,
            ssm_rows,
            has_initial_state,
            verify_update_state(k, batch, device),
            verify_width,
        )
        window_rows, window_offsets = verify_window(merged_offsets, k, device)
        mamba2_replay_fold(
            conv_pool=pools[0],
            ssm_pool=pools[1],
            conv_rows=conv_rows,
            ssm_rows=ssm_rows,
            replay_bounds=replay_plan(window_offsets, num_accepted, k, device),
            merged_offsets=merged_offsets,
            verify_width=verify_width,
            projected=[ops.gather(projected, window_rows, axis=0)],
            convolved=[ops.gather(convolved, window_rows, axis=0)],
            A=[A],
            dt_bias=[dt_bias],
        )

        # The reference arm: main's ops over exactly the committed rows,
        # every request resuming and updating.
        ref_y = []
        for name in range(2):
            kept, kept_offsets, ref_conv_rows, ref_ssm_rows = groups[
                4 * name : 4 * name + 4
            ]
            always = ops.constant(True, DType.bool, device=device).broadcast_to(
                [ref_conv_rows.shape[0]]
            )
            _, out = _layer_forward(
                ops.gather(projected, kept, axis=0),
                conv_weight,
                conv_bias,
                A,
                dt_bias,
                D,
                pools[2],
                pools[3],
                kept_offsets,
                ref_conv_rows,
                ref_ssm_rows,
                always,
                always,
            )
            ref_y.append(out)

        # The decode-only arm: the decode rows alone, with the flags a
        # decode-only step passes.
        alone_rows, alone_offsets, alone_conv_rows, alone_ssm_rows = groups[
            8:12
        ]
        alone_batch = alone_conv_rows.shape[0]
        alone_has = ops.constant(True, DType.bool, device=device).broadcast_to(
            [alone_batch]
        )
        _, alone_y = _layer_rows(
            ops.gather(projected, alone_rows, axis=0),
            conv_weight,
            conv_bias,
            A,
            dt_bias,
            D,
            pools[4],
            pools[5],
            alone_offsets,
            alone_conv_rows,
            alone_ssm_rows,
            alone_has,
            verify_update_state(k, alone_batch, device),
            verify_width,
        )
        graph.output(
            ops.gather(y, kept_rows, axis=0),
            *ref_y,
            ops.gather(y, alone_rows, axis=0),
            alone_y,
        )
    return graph


def _bf16(array: np.ndarray, device: md.Device) -> md.Buffer:
    return md.Buffer.from_dlpack(
        torch.from_numpy(array).to(torch.bfloat16).contiguous()
    ).to(device)


@pytest.fixture(scope="module")
def model() -> Model:
    """Compiles every arm once for all cases."""
    session = InferenceSession(devices=[Accelerator()])
    return session.load(_build_graph(DeviceRef.GPU()))


# Each request is (input rows, accepted drafts), or (input rows, None)
# when its drafts are all MAGIC; the graph reads only the drafts' width.
_CASES: dict[str, tuple[int, list[tuple[int, int | None]]]] = {
    # A prefill row beside decode rows accepting none, part and all of
    # their drafts, and a placeholder decode row that replays its one row.
    "mixed": (3, [(37, None), (1, 2), (1, 0), (1, 3), (1, None)]),
    # Commit points at or below the conv window, where the missing taps
    # come from the stored window.
    "short_prefill": (3, [(2, None), (3, None), (1, 2)]),
    # A prefill row above the chunked floor, between two decode rows.
    "long_chunk": (3, [(1, 3), (700, None), (1, 1)]),
    # A context step: no drafts, so every row commits in the verify.
    "no_drafts": (0, [(20, None), (5, None)]),
    # A decode-only step at a wider runtime draft dim.
    "narrowed": (5, [(1, 0), (1, 3), (1, 5)]),
}


@pytest.mark.skipif(accelerator_count() == 0, reason="Requires GPU")
@pytest.mark.parametrize("case", sorted(_CASES))
def test_mixed_verify_commits_by_row_rule(model: Model, case: str) -> None:
    """Pools and committed outputs match a forward over the committed rows."""
    k, requests = _CASES[case]
    device = model.input_devices[0]
    rng = np.random.default_rng(7)
    batch = len(requests)

    input_lens = [rows for rows, _ in requests]
    merged = np.cumsum([0, *(rows + k for rows in input_lens)])
    drafts = np.asarray(
        [
            [MAGIC_DRAFT_TOKEN_ID] * k
            if accepted is None
            else list(range(100, 100 + k))
            for _, accepted in requests
        ],
        dtype=np.int64,
    ).reshape(batch, k)
    accepted = np.asarray([a or 0 for _, a in requests], dtype=np.uint32)
    # The row rule: only a row with more than one input row is a prefill
    # row; the reference arm runs each kind's committed rows.
    prefill = [b for b, (rows, _) in enumerate(requests) if rows > 1]
    decode = [b for b, (rows, _) in enumerate(requests) if rows == 1]
    pref_lengths = [input_lens[b] for b in prefill]
    dec_lengths = [1 + int(accepted[b]) for b in decode]

    conv_pool = rng.standard_normal((_SLOTS, _CONV_DIM, _WIDTH - 1)).astype(
        np.float32
    )
    ssm_pool = rng.standard_normal(
        (_SLOTS, _NHEADS, _HEAD_DIM, _DSTATE)
    ).astype(np.float32)
    slots = rng.permutation(_SPARE)[: 2 * batch].astype(np.uint32)
    conv_slots, ssm_slots = slots[:batch], slots[batch:]

    def rows_of(members: list[int], lengths: list[int]) -> np.ndarray:
        """One group's global row indices for its ragged launch."""
        return np.concatenate(
            [
                merged[b] + np.arange(n)
                for b, n in zip(members, lengths, strict=True)
            ]
        ).astype(np.int64)

    def ragged(members: list[int], lengths: list[int]) -> list[np.ndarray]:
        """One ragged launch's rows, offsets and pool rows.

        An empty group runs one row into the spare slot, which nothing
        compares.
        """
        if members:
            rows = rows_of(members, lengths)
            conv = conv_slots[members]
            ssm = ssm_slots[members]
        else:
            rows = np.zeros(1, dtype=np.int64)
            lengths = [1]
            conv = ssm = np.full(1, _SPARE, dtype=np.uint32)
        return [
            rows,
            np.cumsum([0, *lengths]).astype(np.int32),
            np.asarray(conv, dtype=np.uint32),
            np.asarray(ssm, dtype=np.uint32),
        ]

    def committed(members: list[int], lengths: list[int]) -> np.ndarray:
        """The committed rows of one group, empty for an empty group."""
        if not members:
            return np.zeros(0, dtype=np.int64)
        return rows_of(members, lengths)

    pref = ragged(prefill, pref_lengths)
    dec = ragged(decode, dec_lengths)
    alone = ragged(decode, [1 + k for _ in decode])
    kept_rows = np.concatenate(
        [committed(prefill, pref_lengths), committed(decode, dec_lengths)]
    )

    pools = [
        _bf16(conv_pool, device),
        md.Buffer.from_numpy(ssm_pool.copy()).to(device),
        _bf16(conv_pool, device),
        md.Buffer.from_numpy(ssm_pool.copy()).to(device),
        _bf16(conv_pool, device),
        md.Buffer.from_numpy(ssm_pool.copy()).to(device),
    ]
    projected = rng.standard_normal((int(merged[-1]), _PROJECTED)).astype(
        np.float32
    )
    y, pref_y, dec_y, full_y, alone_y = model.execute(
        *pools,
        md.Buffer.from_numpy(conv_slots.reshape(1, batch).copy()).to(device),
        md.Buffer.from_numpy(ssm_slots.reshape(1, batch).copy()).to(device),
        md.Buffer.from_numpy(drafts).to(device),
        md.Buffer.from_numpy(merged.astype(np.uint32)).to(device),
        md.Buffer.from_numpy(accepted).to(device),
        _bf16(projected, device),
        md.Buffer.from_numpy(kept_rows).to(device),
        *[
            md.Buffer.from_numpy(a).to(device)
            for g in (pref, dec, alone)
            for a in g
        ],
        _bf16(-(rng.random(_NHEADS, dtype=np.float32) + 0.1), device),
        _bf16(rng.random(_NHEADS, dtype=np.float32), device),
        _bf16(rng.random(_CONV_DIM, dtype=np.float32), device),
        _bf16(rng.random(_NHEADS, dtype=np.float32), device),
        _bf16(
            rng.standard_normal((_CONV_DIM, _WIDTH)).astype(np.float32), device
        ),
    )
    verify_conv, verify_ssm, ref_conv, ref_ssm, _, _ = (
        torch.from_dlpack(p).cpu().float().numpy() for p in pools
    )
    # The conv window is the last committed inputs themselves.
    np.testing.assert_array_equal(verify_conv[:_SPARE], ref_conv[:_SPARE])
    # A prefill row commits in the verify, a decode row in the fold, and
    # both land the state a forward over the committed rows would.
    np.testing.assert_array_equal(verify_ssm[:_SPARE], ref_ssm[:_SPARE])
    np.testing.assert_array_equal(
        torch.from_dlpack(y).cpu().float().numpy(),
        np.concatenate(
            [
                torch.from_dlpack(r).cpu().float().numpy()[:n]
                for r, n in (
                    (pref_y, sum(pref_lengths)),
                    (dec_y, sum(dec_lengths)),
                )
            ]
        ),
    )
    # A decode row computes the same in a mixed step as in a decode-only
    # step.
    if decode:
        np.testing.assert_array_equal(
            torch.from_dlpack(full_y).cpu().float().numpy(),
            torch.from_dlpack(alone_y).cpu().float().numpy(),
        )
    # Every request committed something, so every request's state moved.
    for slot in ssm_slots:
        assert not np.array_equal(verify_ssm[slot], ssm_pool[slot])
    outside = [s for s in range(_SLOTS) if s not in set(slots.tolist())]
    conv_seed = torch.from_numpy(conv_pool).to(torch.bfloat16).float().numpy()
    np.testing.assert_array_equal(verify_ssm[outside], ssm_pool[outside])
    np.testing.assert_array_equal(verify_conv[outside], conv_seed[outside])
