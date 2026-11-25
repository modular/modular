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
"""The head-free draft score: the graph output against a float64 reference."""

from __future__ import annotations

import numpy as np
from max import mlir
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.pipelines.speculative.draft_scores import (
    combine_shard_scores,
    top_token_logit,
    top_token_pieces,
)
from max.tensor import Tensor
from max.tensor.realization_context import ensure_context
from speculative_test_helpers import (
    check_batched_combine,
    reference_scores,
    run_sharded_scores,
    shard_maxima_and_indices,
)
from speculative_test_helpers import cpu_session as cpu_session


def test_the_score_and_pieces_match_the_reference(
    session: InferenceSession,
) -> None:
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(2, 3, 17)).astype(np.float32)
    logits[0, 1] = 0.0
    logits[1, 0] = -15.0
    logits[1, 0, 0] = 0.0
    logits[1, 1, [4, 9]] = 6.0
    start = 5
    device = DeviceRef.CPU()
    input_type = TensorType(DType.float32, list(logits.shape), device)
    with Graph("score_and_pieces", input_types=[input_type]) as graph:
        values = graph.inputs[0].tensor
        graph.output(top_token_logit(values), top_token_pieces(values, start))
    score_buf, pieces_buf = session.load(graph).execute(
        Buffer.from_numpy(logits)
    )
    scores, pieces = score_buf.to_numpy(), pieces_buf.to_numpy()
    assert scores.shape == logits.shape[:-1]
    np.testing.assert_allclose(
        scores, reference_scores(logits), rtol=1e-5, atol=1e-5
    )
    assert scores[1, 0] > 10.0
    np.testing.assert_allclose(
        scores[0, 1], -np.log(logits.shape[-1] - 1), atol=1e-4
    )
    assert pieces.shape == (*scores.shape, 3)
    np.testing.assert_allclose(
        pieces[..., 0], logits.max(axis=-1), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_array_equal(
        pieces[..., 1], logits.argmax(axis=-1) + start
    )
    np.testing.assert_allclose(
        pieces[..., 2], np.exp(-scores.astype(np.float64)), rtol=1e-5, atol=1e-5
    )


def test_hyper_peaked_logits_keep_the_deficit(
    session: InferenceSession,
) -> None:
    two = np.array([[[0.0, -16.4]], [[0.0, -17.0]]], dtype=np.float32)
    vector = np.array([0.0, -0.5, -1.0], dtype=np.float32)
    device = DeviceRef.CPU()
    with Graph(
        "two_token_scores",
        input_types=[
            TensorType(DType.float32, list(two.shape), device),
            TensorType(DType.float32, list(vector.shape), device),
        ],
    ) as graph:
        graph.output(*(top_token_logit(value.tensor) for value in graph.inputs))
    rows, scalar = session.load(graph).execute(
        Buffer.from_numpy(two), Buffer.from_numpy(vector)
    )
    np.testing.assert_allclose(rows.to_numpy()[:, 0], [16.4, 17.0], atol=1e-3)
    assert scalar.to_numpy().shape == ()
    np.testing.assert_allclose(
        scalar.to_numpy(), reference_scores(vector), rtol=1e-5, atol=1e-5
    )


def test_sharded_combine_keeps_ties_and_peaked_scores(
    session: InferenceSession,
) -> None:
    logits = np.random.default_rng(7).normal(size=(4, 6)).astype(np.float32)
    logits[0, [0, 2]] = 5.0
    logits[1, [0, 4]] = 5.0
    logits[3] = [-60.0, 0.0, -60.0, -60.0, -25.0, -60.0]
    scores = run_sharded_scores(session, logits, [3, 3])
    np.testing.assert_allclose(
        scores, reference_scores(logits), rtol=1e-5, atol=1e-5
    )


def test_sharded_combine_adds_nothing_for_empty_shards(
    session: InferenceSession,
) -> None:
    logits = np.full((2, 6), -np.inf, dtype=np.float32)
    logits[1, :3] = [1.0, 3.0, 2.0]
    scores = run_sharded_scores(session, logits, [3, 3])
    assert np.isposinf(scores[0]), scores
    np.testing.assert_allclose(
        scores[1], reference_scores(logits[1, :3]), rtol=1e-5, atol=1e-5
    )


def test_the_score_builds_the_fused_op() -> None:
    # The score must run through the one fused kernel, not the old
    # three-reduction body.
    device = DeviceRef.CPU()
    input_type = TensorType(DType.float32, [2, 3, 8], device)
    with Graph("top_token_logit", input_types=[input_type]) as graph:
        graph.output(top_token_logit(graph.inputs[0].tensor))
    symbols = [
        mlir.StringAttr(op.attributes["symbol"]).value
        for op in graph._mlir_op.regions[0].blocks[0].operations
        if op.name == "mo.custom"
    ]
    assert "spec.top_token_logit" in symbols, symbols


def test_the_pieces_build_the_fused_op() -> None:
    # The pieces must run through the one fused kernel, not the old
    # unfused ops.
    device = DeviceRef.CPU()
    input_type = TensorType(DType.float32, [2, 3, 8], device)
    with Graph("top_token_pieces", input_types=[input_type]) as graph:
        graph.output(top_token_pieces(graph.inputs[0].tensor, 11))
    symbols = [
        mlir.StringAttr(op.attributes["symbol"]).value
        for op in graph._mlir_op.regions[0].blocks[0].operations
        if op.name == "mo.custom"
    ]
    assert "spec.top_token_pieces" in symbols, symbols


def test_the_batched_combine_matches_the_per_position_combine(
    session: InferenceSession,
) -> None:
    rng = np.random.default_rng(20261118)
    widths = [2, 3, 2, 2]
    b, k = 2, 3
    logits = rng.normal(size=(b, k, sum(widths))).astype(np.float32)
    shards = np.split(logits, np.cumsum(widths)[:-1], axis=-1)
    maxima, indices = shard_maxima_and_indices(shards)
    masses = np.stack([np.exp(-reference_scores(shard)) for shard in shards])
    pieces = np.stack([maxima, indices, masses], axis=-1).astype(np.float32)
    top_indices = logits.argmax(axis=-1).astype(np.int32)
    top_max = logits.max(axis=-1)
    check_batched_combine(session, pieces, top_indices, top_max)


def test_the_batched_combine_runs_one_reduction_set() -> None:
    """The batched combine reduces the shard axis once for the whole
    block, not once per position, and moves no axis to do it."""
    device = DeviceRef.CPU()
    shards_n, b, k = 4, 5, 3
    pieces_type = TensorType(DType.float32, [shards_n, b, k, 3], device)
    ids_type = TensorType(DType.int32, [b, k], device)
    top_type = TensorType(DType.float32, [b, k], device)
    with (
        Graph(
            "combine_shard_scores_once",
            input_types=[pieces_type, ids_type, top_type],
        ) as graph,
        ensure_context(),
    ):
        graph.output(
            combine_shard_scores(
                Tensor.from_graph_value(graph.inputs[0].tensor),
                Tensor.from_graph_value(graph.inputs[1].tensor),
                Tensor.from_graph_value(graph.inputs[2].tensor),
            )
        )
    names = [op.name for op in graph._mlir_op.regions[0].blocks[0].operations]
    # Two shard-axis sums and one exp and log each: the per-position
    # path would run k times as many.
    assert names.count("mo.reduce.add") == 2, names
    assert names.count("mo.exp") == 1, names
    assert names.count("mo.log") == 1, names
    # A transpose would add its own copy kernel.
    assert not any(
        "transpose" in name or "permute" in name for name in names
    ), names
