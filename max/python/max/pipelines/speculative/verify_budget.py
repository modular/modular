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
"""Per-row verify lengths from calibrated draft survival."""

from __future__ import annotations

from max.dtype import DType
from max.graph import DeviceRef, Dim, TensorValue, ops
from max.pipelines.speculative.config import MAGIC_DRAFT_TOKEN_ID
from max.pipelines.speculative.ragged_token_merger import _shape_to_scalar

__all__ = [
    "SCORE_LIMIT",
    "draft_survival",
    "floored_survival",
    "rows_without_proposals",
    "select_replica_verify_lengths",
    "select_verify_lengths",
    "unverified_draft_placeholders",
]

# Must equal SCORE_LIMIT in mach/crates/engine-sched/src/spec_calibrator.rs;
# the host fit skips a NaN score where the graph bounds it.
SCORE_LIMIT = 64.0


def draft_survival(scores: TensorValue, coeffs: TensorValue) -> TensorValue:
    """Computes each draft position's survival from its score.

    The running product of the per-position acceptance probabilities the
    calibrator's coefficients give, so a later position's survival never
    exceeds an earlier one's.

    Args:
        scores: Draft scores, ``[B, K]`` float32.
        coeffs: The calibrator's coefficients, ``[K + 1]`` float32: the
            shared slope first, then one intercept per position.

    Returns:
        Survival, ``[B, K]`` float32.
    """
    # The clamp maps a NaN score to the low bound (IEEE maxNum/minNum),
    # so a NaN draft is never chosen.
    bounded = ops.min(
        ops.max(
            scores,
            ops.constant(-SCORE_LIMIT, DType.float32, device=scores.device),
        ),
        ops.constant(SCORE_LIMIT, DType.float32, device=scores.device),
    )
    slope = coeffs[0]
    intercepts = ops.reshape(coeffs[1:], [1, coeffs.shape[0] - 1])
    logits = bounded * slope + intercepts
    # The product of probabilities as a cumsum of logs, so the running
    # product never underflows a float32 multiply.
    log_probs = ops.log(ops.sigmoid(logits))
    return ops.exp(ops.cumsum(log_probs, axis=1))


def floored_survival(
    scores: TensorValue,
    coeffs: TensorValue,
    no_proposals: TensorValue,
) -> TensorValue:
    """Survival with a no-proposal row's entries floored below every real one.

    A no-proposal row carries no score to rank, so its floor sits
    strictly below every real survival, even one that underflows. The
    negative floor also marks the row for the selection, which gives it
    no verify slot.
    """
    survival = draft_survival(scores, coeffs)
    return ops.where(
        ops.unsqueeze(no_proposals, axis=-1),
        ops.constant(-1.0, survival.dtype, device=survival.device),
        survival,
    )


def rows_without_proposals(
    draft_tokens: TensorValue,
    num_steps: Dim,
    device: DeviceRef,
    *,
    padded: bool,
) -> TensorValue:
    """Flags each row whose every bound draft is the no-proposal fill.

    The selection and the accept both need the flag, on drafts whose width
    is the bound ``K`` rather than the step's verify width.

    Args:
        draft_tokens: Draft token ids, ``[B, K]`` int64.
        num_steps: The bound draft width ``K``.
        device: The device the flag is built on.
        padded: Whether the driver pads the drafts with one MAGIC
            column; the sequential driver does, the block driver does
            not, and each driver's flag must match its own form.
    """
    if padded:
        # The extra MAGIC column makes the reduction well-defined at K == 0.
        magic_token = ops.constant(MAGIC_DRAFT_TOKEN_ID, DType.int64, device)
        padded_drafts = ops.concat(
            [draft_tokens, magic_token.broadcast_to(["batch_size", 1])],
            axis=1,
        )
        num_magic_tokens = ops.squeeze(
            ops.sum((padded_drafts == magic_token).cast(DType.int32), axis=-1),
            axis=-1,
        )
        num_steps_plus_one = (
            _shape_to_scalar(num_steps, device, DType.int32) + 1
        )
        return num_magic_tokens == num_steps_plus_one.broadcast_to(
            ["batch_size"]
        )
    num_magic_tokens = ops.squeeze(
        ops.sum(
            (draft_tokens == MAGIC_DRAFT_TOKEN_ID)
            .cast(DType.int32)
            .rebind(["batch_size", "num_steps"]),
            axis=-1,
        ),
        axis=-1,
    )
    num_steps_i32 = _shape_to_scalar(num_steps, device, DType.int32)
    return num_magic_tokens == num_steps_i32.broadcast_to(["batch_size"])


def select_verify_lengths(
    survival: TensorValue,
    total: TensorValue,
    replica_ids: TensorValue | None = None,
) -> TensorValue:
    """Splits a fixed verify budget over rows by survival.

    Keeps the ``total`` entries with the highest survival among the
    ranked entries, ordered by (value desc, position asc, row asc), and
    returns how many each row kept. An entry whose survival is negative
    is a no-proposal row (see ``floored_survival``): it never enters the
    rank and keeps no slot. With ``replica_ids`` set, ``total`` is the
    per-row budget (each replica's rows times its width) and the rank
    counts only same-replica entries, so each replica spends its own
    budget; the whole-batch entry is ``select_replica_verify_lengths``.
    This is the graph twin of the test reference ``top_t_lengths`` in
    ``engine-sched``'s ``spec_budget.rs``.

    Args:
        survival: Survival, ``[B, K]`` float32 with static ``K``;
            ``B`` may be symbolic, each row non-increasing.
        total: The budget, a scalar int32 tensor on the same device,
            or a per-row ``[B]`` budget with ``replica_ids`` set.
        replica_ids: Per-row replica ids, ``[B]`` int32 on the same
            device, or ``None`` to rank the whole batch together.

    Returns:
        Per-row verify lengths, ``[B]`` int32, summing to at most
        ``min(total, B * K)`` without ``replica_ids``, or to at most
        ``min(total, B_r * K)`` per replica with it; a no-proposal row
        keeps none.
    """
    batch = survival.shape[0]
    positions = int(survival.shape[1])
    flat_size = batch * positions
    flat = ops.reshape(survival, [flat_size])
    device = survival.device
    # A real survival is a product of probabilities, so a negative entry is
    # ``floored_survival``'s no-proposal marker.
    unranked = flat < ops.constant(0.0, flat.dtype, device=device)
    index = ops.range(
        start=0,
        stop=ops.shape_to_tensor([flat_size])[0],
        out_dim=flat_size,
        device=device,
        dtype=DType.int32,
    )
    # The flat index encodes both tie keys: position first, then row.
    pos = ops.reshape(index % positions, [flat_size, 1])
    row = ops.reshape(index // positions, [flat_size, 1])
    values = ops.reshape(flat, [flat_size, 1])
    pos_i = ops.broadcast_to(pos, [flat_size, flat_size])
    row_i = ops.broadcast_to(row, [flat_size, flat_size])
    values_i = ops.broadcast_to(values, [flat_size, flat_size])
    # Quadratic in B * K; max_batch_size x num_draft_steps bounds it.
    beats = (values_i > values.T) | (
        (values_i == values.T)
        & ((pos_i < pos.T) | ((pos_i == pos.T) & (row_i < row.T)))
    )
    if replica_ids is not None:
        # A cross-replica entry never competes for this replica's
        # budget, so it must not enter the rank.
        flat_ids = ops.reshape(
            ops.broadcast_to(
                ops.reshape(replica_ids, [batch, 1]),
                [batch, positions],
            ),
            [flat_size, 1],
        )
        flat_ids = ops.broadcast_to(flat_ids, [flat_size, flat_size])
        beats = beats & (flat_ids == flat_ids.T)
    # An unranked entry never beats a ranked one, so it cannot consume a rank
    # position the ranked entries compete for.
    beats = beats & ops.logical_not(
        ops.broadcast_to(
            ops.reshape(unranked, [flat_size, 1]), [flat_size, flat_size]
        )
    )
    rank = ops.sum(ops.cast(beats, DType.int32), axis=0)
    if replica_ids is not None:
        # The budget is per row (its replica's own), so it broadcasts
        # down each row's positions.
        kept = ops.reshape(rank, [batch, positions]) < ops.reshape(
            total, [batch, 1]
        )
    else:
        kept = ops.reshape(rank < total, [batch, positions])
    # A no-proposal row verifies nothing, whatever its rank was.
    ranked = ops.reshape(ops.logical_not(unranked), [batch, positions])
    return ops.squeeze(
        ops.sum(ops.cast(kept & ranked, DType.int32), axis=1), axis=1
    )


def select_replica_verify_lengths(
    survival: TensorValue,
    width: TensorValue,
    data_parallel_splits: TensorValue | None,
) -> TensorValue:
    """Splits each DP replica's verify budget over its own rows.

    Args:
        survival: Survival, ``[B, K]`` float32 with static ``K``; ``B``
            may be symbolic. The whole batch, on one device.
        width: The step's uniform verify width, a scalar int32 tensor on
            the survival's device.
        data_parallel_splits: The CPU int64 ``[dp + 1]`` row boundaries
            input, or ``None`` without data parallelism.

    Returns:
        Per-row verify lengths, ``[B]`` int32 in batch order; each
        replica's lengths sum to at most ``min(B_r * width, B_r * K)``,
        and a no-proposal row keeps none.
    """
    batch = survival.shape[0]
    device = survival.device
    if data_parallel_splits is None or int(data_parallel_splits.shape[0]) == 2:
        # One replica owns every row, so no mask is needed and the
        # splits input stays unread on the device.
        return select_verify_lengths(
            survival,
            _shape_to_scalar(batch, device, dtype=DType.int32) * width,
        )
    replicas = int(data_parallel_splits.shape[0]) - 1
    # Host boundaries reach the device as one-element ranges, safe because
    # a captured step pads replicas uniformly, so splits follow the batch.
    boundaries = [
        ops.range(
            start=data_parallel_splits[k],
            stop=data_parallel_splits[k] + 1,
            out_dim=1,
            device=device,
            dtype=DType.int32,
        )[0]
        for k in range(1, replicas)
    ]
    row_index = ops.range(
        start=0,
        stop=ops.shape_to_tensor([batch])[0],
        out_dim=batch,
        device=device,
        dtype=DType.int32,
    )
    # Row i belongs to the replica whose interior boundaries all sit
    # at or below i; a zero-row replica contributes no boundary.
    replica_ids = ops.squeeze(
        ops.sum(
            ops.cast(
                ops.unsqueeze(row_index, axis=0)
                >= ops.unsqueeze(ops.stack(boundaries), axis=1),
                DType.int32,
            ),
            axis=0,
        ),
        axis=0,
    )
    # Budget each row by its own replica's row count, so a replica never
    # spends another's slots even when the splits are non-uniform.
    splits_full = ops.concat(
        [
            ops.constant(0, DType.int32, device).broadcast_to([1]),
            *(ops.unsqueeze(boundary, 0) for boundary in boundaries),
            _shape_to_scalar(batch, device, dtype=DType.int32).broadcast_to(
                [1]
            ),
        ],
        axis=0,
    )
    starts = ops.gather(splits_full, replica_ids, axis=0)
    ends = ops.gather(splits_full, replica_ids + 1, axis=0)
    budget = (ends - starts) * width
    return select_verify_lengths(survival, budget, replica_ids)


def unverified_draft_placeholders(
    draft_tokens: TensorValue, verify_lengths: TensorValue
) -> TensorValue:
    """Masks each row's unverified draft positions to a never-matching id.

    The masked id equals no target token, so the sampler's argmax verdicts
    reject those positions; only the sampler reads this copy.

    Args:
        draft_tokens: Draft token ids, ``[B, K]`` int32 or int64.
        verify_lengths: Per-row verify lengths, ``[B]`` int32.

    Returns:
        The masked copy, ``[B, K]`` in the input's dtype.
    """
    positions = draft_tokens.shape[1]
    index = ops.range(
        start=0,
        stop=ops.shape_to_tensor([positions])[0],
        out_dim=positions,
        device=draft_tokens.device,
        dtype=draft_tokens.dtype,
    )
    past = ops.unsqueeze(index, 0) >= ops.cast(
        ops.unsqueeze(verify_lengths, 1), draft_tokens.dtype
    )
    return ops.where(past, -1, draft_tokens)
