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
"""Shared graph fixtures and references for speculative pipeline tests."""

from __future__ import annotations

import itertools

import numpy as np
from cpu_step_session import cpu_session as cpu_session
from max import functional as F
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, ShapeLike, TensorType, ops
from max.pipelines.speculative.draft_scores import (
    combine_shard_scores,
    top_token_logit,
    top_token_pieces,
)
from max.pipelines.speculative.verify_budget import (
    SCORE_LIMIT,
    floored_survival,
    rows_without_proposals,
    select_replica_verify_lengths,
    select_verify_lengths,
    unverified_draft_placeholders,
)
from max.tensor import Tensor
from max.tensor.realization_context import ensure_context


def reference_scores(logits: np.ndarray) -> np.ndarray:
    """Computes the top token's log-odds in float64."""
    wide = logits.astype(np.float64)
    top = wide.max(axis=-1, keepdims=True)
    gap = -np.log(np.exp(wide - top).sum(axis=-1))
    return gap - np.log1p(-np.exp(gap))


def shard_maxima_and_indices(
    shards: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Finds each shard's row maxima and global token indices.

    Args:
        shards: Logits in vocabulary order, with vocabulary on the last axis.

    Returns:
        Float32 maxima and indices stacked along the shard axis.
    """
    starts = np.cumsum([0] + [s.shape[-1] for s in shards[:-1]])
    maxima = np.stack([s.max(axis=-1) for s in shards], axis=0).astype(
        np.float32
    )
    indices = np.stack(
        [
            s.argmax(axis=-1) + start
            for s, start in zip(shards, starts, strict=True)
        ],
        axis=0,
    ).astype(np.float32)
    return maxima, indices


def check_batched_combine(
    session: InferenceSession,
    pieces: np.ndarray,
    top_indices: np.ndarray,
    top_max: np.ndarray,
) -> None:
    """Checks that batched shard scores match each position bit for bit.

    Args:
        session: CPU inference session for both combine graphs.
        pieces: Float32 maxima, global indices and excluded masses, indexed
            by shard, row and position.
        top_indices: Int32 winning token indices by row and position.
        top_max: Float32 winning logits by row and position.
    """
    shards_n, b, k, _ = pieces.shape
    device = DeviceRef.CPU()

    def combine(width: int) -> Model:
        input_types = [
            TensorType(DType.float32, [shards_n, b, width, 3], device),
            TensorType(DType.int32, [b, width], device),
            TensorType(DType.float32, [b, width], device),
        ]
        with (
            Graph(f"combine_{width}", input_types=input_types) as graph,
            ensure_context(),
        ):
            graph.output(
                combine_shard_scores(
                    Tensor.from_graph_value(graph.inputs[0].tensor),
                    Tensor.from_graph_value(graph.inputs[1].tensor),
                    Tensor.from_graph_value(graph.inputs[2].tensor),
                )
            )
        return session.load(graph)

    one = combine(1)
    want = np.zeros((b, k), dtype=np.float32)
    for i in range(k):
        (out,) = one.execute(
            Buffer.from_numpy(np.ascontiguousarray(pieces[:, :, i : i + 1])),
            Buffer.from_numpy(np.ascontiguousarray(top_indices[:, i : i + 1])),
            Buffer.from_numpy(np.ascontiguousarray(top_max[:, i : i + 1])),
        )
        want[:, i] = out.to_numpy()[:, 0]

    (out,) = combine(k).execute(
        Buffer.from_numpy(pieces),
        Buffer.from_numpy(top_indices),
        Buffer.from_numpy(top_max),
    )

    actual = out.to_numpy()
    assert actual.shape == (b, k)
    np.testing.assert_array_equal(actual.view(np.uint32), want.view(np.uint32))


def run_sharded_scores(
    session: InferenceSession,
    logits: np.ndarray,
    widths: list[int],
) -> np.ndarray:
    """Runs the shard combine and checks it against the joined logits."""
    starts = np.cumsum([0, *widths[:-1]])
    shards = np.split(logits, starts[1:], axis=-1)
    device = DeviceRef.CPU()
    batch = logits.shape[0]
    input_types = [
        TensorType(DType.float32, [batch, width], device) for width in widths
    ]
    input_types.extend(
        [
            TensorType(DType.int32, [batch, 1], device),
            TensorType(DType.float32, [batch, 1], device),
        ]
    )
    with (
        Graph("combine_shard_scores", input_types=input_types) as graph,
        ensure_context(),
    ):
        values = [v.tensor for v in graph.inputs[:-2]]
        pieces = [
            top_token_pieces(value, int(start))
            for value, start in zip(values, starts, strict=True)
        ]
        combined = combine_shard_scores(
            F.unsqueeze(F.stack(pieces, axis=0), 2),
            Tensor.from_graph_value(graph.inputs[-2].tensor),
            Tensor.from_graph_value(graph.inputs[-1].tensor),
        )
        graph.output(
            F.squeeze(combined, axis=-1),
            top_token_logit(ops.concat(values, axis=-1)),
        )
    sharded, joined = session.load(graph).execute(
        *(Buffer.from_numpy(np.ascontiguousarray(s)) for s in shards),
        Buffer.from_numpy(
            logits.argmax(axis=-1, keepdims=True).astype(np.int32)
        ),
        Buffer.from_numpy(logits.max(axis=-1, keepdims=True)),
    )
    scores = sharded.to_numpy()
    assert scores.shape == (batch,)
    np.testing.assert_allclose(scores, joined.to_numpy(), rtol=1e-5, atol=1e-5)
    return scores


def build_lengths_model(session: InferenceSession, shape: ShapeLike) -> Model:
    """Loads a selection graph for the given batch and draft width.

    Args:
        session: The CPU inference session.
        shape: The survival matrix shape, with a fixed or symbolic batch size
            and a static draft width.

    Returns:
        The model mapping survival scores and a scalar budget to row lengths.
    """
    device = DeviceRef.CPU()
    with Graph(
        "select_verify_lengths",
        input_types=[
            TensorType(DType.float32, shape, device),
            TensorType(DType.int32, [], device),
        ],
    ) as graph:
        graph.output(
            select_verify_lengths(
                graph.inputs[0].tensor, graph.inputs[1].tensor
            )
        )
    return session.load(graph)


def build_floored_lengths_model(session: InferenceSession) -> Model:
    """Loads the floor, replica selection and placeholder graph.

    Args:
        session: The CPU inference session.

    Returns:
        The model returning floored survival scores, selected lengths and
        masked drafts for two replicas at a static draft width of four.
    """
    k = 4
    device = DeviceRef.CPU()
    input_types = [
        TensorType(DType.int64, ["batch_size", "num_steps"], device),
        TensorType(DType.float32, ["batch_size", k], device),
        TensorType(DType.float32, [k + 1], device),
        TensorType(DType.int32, [], device),
        TensorType(DType.int64, [3], device),
    ]
    with Graph("floored_verify_lengths", input_types=input_types) as g:
        drafts, scores, coeffs, width, splits = (x.tensor for x in g.inputs)
        no_proposals = rows_without_proposals(
            drafts, drafts.shape[1], device, padded=True
        )
        survival = floored_survival(scores, coeffs, no_proposals)
        lengths = select_replica_verify_lengths(survival, width, splits)
        g.output(
            survival, lengths, unverified_draft_placeholders(drafts, lengths)
        )
    return session.load(g)


def reference_survival(scores: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """Computes cumulative survival probabilities in float64."""
    wide = scores.astype(np.float64)
    # The kernel's min/max drop NaN (IEEE maxNum/minNum), so the
    # reference maps it to the low bound the same way.
    wide = np.where(np.isnan(wide), -SCORE_LIMIT, wide)
    wide = np.clip(wide, -SCORE_LIMIT, SCORE_LIMIT)
    logits = wide * coeffs[0].astype(np.float64) + coeffs[1:].astype(np.float64)
    probs = 1.0 / (1.0 + np.exp(-logits))
    return np.cumprod(probs, axis=1)


def reference_lengths(
    survival: np.ndarray, total: int
) -> tuple[np.ndarray, np.ndarray]:
    """Returns the kept positions and row lengths in budget rank order."""
    batch, positions = survival.shape
    flat = survival.reshape(-1)
    order = np.lexsort(
        (
            np.arange(batch * positions) // positions,
            np.arange(batch * positions) % positions,
            -flat,
        )
    )
    kept = np.zeros(batch * positions, dtype=bool)
    kept[order[: min(total, batch * positions)]] = True
    return kept.reshape(batch, positions), kept.reshape(batch, positions).sum(
        axis=1
    )


def sorted_survival(rng: np.random.Generator, batch: int, k: int) -> np.ndarray:
    """Generates non-increasing survival rows."""
    return np.sort(rng.random((batch, k)), axis=1)[:, ::-1].astype(np.float32)


# Higher-scoring rows would take slots from other replicas in a global rank.
RANKED_SURVIVAL = np.array(
    [
        [0.9, 0.8, 0.8],
        [0.85, 0.8, 0.75],
        [0.7, 0.65, 0.6],
        [0.6, 0.55, 0.5],
        [0.5, 0.5, 0.4],
        [0.5, 0.4, 0.3],
        [0.3, 0.25, 0.2],
        [0.2, 0.1, 0.1],
    ],
    dtype=np.float32,
)


def _replica_reference(
    survival: np.ndarray, splits: np.ndarray, width: int
) -> np.ndarray:
    """Computes reference lengths within each replica's budget."""
    lengths = []
    for start, end in itertools.pairwise(splits):
        _, rows = reference_lengths(survival[start:end], (end - start) * width)
        lengths.append(rows)
    return np.concatenate(lengths)


def build_replica_lengths_model(
    session: InferenceSession, shape: tuple[int, ...], split_count: int
) -> Model:
    """Loads replica selection for a static survival matrix and split count.

    Args:
        session: The CPU inference session.
        shape: The static batch size and draft width of the survival matrix.
        split_count: The number of replica boundaries, one more than the
            number of replicas.

    Returns:
        The model mapping survival scores, width and splits to row lengths.
    """
    device = DeviceRef.CPU()
    with Graph(
        "select_replica_verify_lengths",
        input_types=[
            TensorType(DType.float32, shape, device),
            TensorType(DType.int32, [], device),
            TensorType(DType.int64, [split_count], device),
        ],
    ) as graph:
        scores, width, splits = (x.tensor for x in graph.inputs)
        graph.output(select_replica_verify_lengths(scores, width, splits))
    return session.load(graph)


def run_replica_lengths(
    model: Model, survival: np.ndarray, splits: np.ndarray, width: int
) -> np.ndarray:
    """Runs replica selection with a uniform verify width.

    Args:
        model: Compiled replica selection graph.
        survival: Float32 survival probabilities by batch row and draft position.
        splits: Int64 row boundaries from zero to the batch size, with one
            interval per replica.
        width: Scalar int32 verify slots per row, pooled within each replica.

    Returns:
        An int32 vector of selected prefix lengths in batch order.
    """
    (out,) = model.execute(
        Buffer.from_numpy(survival),
        Buffer.from_numpy(np.array(width, dtype=np.int32)),
        Buffer.from_numpy(splits),
    )
    return out.to_numpy()


def check_replica_budgets(
    session: InferenceSession, cases: list[tuple[np.ndarray, list[int], int]]
) -> None:
    """Checks each replica against its independently ranked rows."""
    survival, boundaries, _ = cases[0]
    model = build_replica_lengths_model(
        session, survival.shape, len(boundaries)
    )
    for survival, boundaries, step_width in cases:
        row_splits = np.array(boundaries, dtype=np.int64)
        lengths = run_replica_lengths(model, survival, row_splits, step_width)
        np.testing.assert_array_equal(
            lengths, _replica_reference(survival, row_splits, step_width)
        )
