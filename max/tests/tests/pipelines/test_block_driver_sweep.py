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
"""Additional block driver CPU cases."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from block_step_helpers import (
    PER_ROW_B,
    PER_ROW_K,
    PER_ROW_W,
    VOCAB,
    build_block_step,
    run_per_row,
)
from cpu_step_session import cpu_session as cpu_session
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType
from max.pipelines.speculative.ragged_token_merger import (
    compute_host_merged_offsets,
)


@pytest.fixture(scope="module")
def sampled_step(session: InferenceSession) -> Model:
    return build_block_step(session, per_row=True, stochastic=True)


@pytest.fixture(scope="module")
def per_row_host_step(session: InferenceSession) -> Model:
    return build_block_step(session, per_row=True, host_mirror=True)


@pytest.fixture(scope="module")
def uniform_host_step(session: InferenceSession) -> Model:
    return build_block_step(session, per_row=False, host_mirror=True)


def test_a_sampled_per_row_verify_rejects_at_each_rows_length(
    sampled_step: Model,
) -> None:
    """Under sampled acceptance a row commits its own sample at k_r, so
    the drafts past k_r never win even when they match the target."""
    prompt_lens = [1, 1]
    drafts = np.tile(np.array([[7, 11, 13, 15]], dtype=np.int64), (2, 1))
    # The budget is two slots (w = 1 over two rows): row 0's top two
    # survivals take both, row 1 keeps none.
    scores = np.array(
        [[5.0, 5.0, -5.0, -5.0], [-5.0, -5.0, -5.0, -5.0]],
        dtype=np.float32,
    )
    coeffs = np.array([1.0] + [0.0] * PER_ROW_K, dtype=np.float32)
    # Every draft position's target argmax matches the draft, so without
    # the per-row mask the sampled verdict would accept past k_r.
    argmax = np.tile(drafts[0], PER_ROW_K + 1).astype(np.int64)
    merged_len = sum(prompt_lens) + 2 * 1
    num_accepted, next_tokens, _drafts, lengths, _, _ = run_per_row(
        sampled_step,
        prompt_lens,
        drafts,
        scores,
        coeffs,
        argmax,
        merged_len,
    )
    np.testing.assert_array_equal(lengths, [2, 0])
    np.testing.assert_array_equal(num_accepted, [2, 0])
    np.testing.assert_array_equal(next_tokens, [13, 11])


def test_a_sampled_per_row_verify_commits_the_sample_at_a_reject(
    sampled_step: Model,
) -> None:
    """A draft the target sample rejects is replaced by that sample, which
    the commit takes even before the row's k_r is spent."""
    prompt_lens = [1, 1]
    drafts = np.tile(np.array([[7, 11, 13, 15]], dtype=np.int64), (2, 1))
    # Row 0's scores keep two slots, but its position-1 draft differs from
    # the target sample there, so the run stops at one.
    scores = np.array(
        [[5.0, 5.0, -5.0, -5.0], [-5.0, -5.0, -5.0, -5.0]],
        dtype=np.float32,
    )
    coeffs = np.array([1.0] + [0.0] * PER_ROW_K, dtype=np.float32)
    # Position 0 matches its draft; position 1 samples a token no draft holds,
    # under the mask token; rows past both rows' verify windows are unread.
    argmax = np.array([7, 9, 13, 11, 13, 11, 11, 11, 11, 11], dtype=np.int64)
    merged_len = sum(prompt_lens) + 2 * 1
    num_accepted, next_tokens, _drafts, lengths, _, _ = run_per_row(
        sampled_step,
        prompt_lens,
        drafts,
        scores,
        coeffs,
        argmax,
        merged_len,
    )
    np.testing.assert_array_equal(lengths, [2, 0])
    np.testing.assert_array_equal(num_accepted, [1, 0])
    np.testing.assert_array_equal(next_tokens, [9, 11])


def test_equal_scores_match_the_uniform_narrow_verify(
    per_row_host_step: Model, uniform_host_step: Model
) -> None:
    """Equal scores give every row k_r = w, and the per-row driver's outputs
    equal the same driver with the flag off, bound at [B, w]."""
    rng = np.random.default_rng(12)
    prompt_lens = [1, 2, 1]
    drafts = rng.integers(200, 300, size=(PER_ROW_B, PER_ROW_K)).astype(
        np.int64
    )
    scores = np.zeros((PER_ROW_B, PER_ROW_K), dtype=np.float32)
    coeffs = np.array([1.0] + [0.0] * PER_ROW_K, dtype=np.float32)
    # One argmax per row, so both runs judge the same drafts; row 0's
    # match one past w, so a dropped or short mask moves the accepted count.
    argmax = rng.integers(0, VOCAB, size=PER_ROW_B)
    drafts[0, : PER_ROW_W + 1] = argmax[0]
    merged_len = sum(prompt_lens) + PER_ROW_B * PER_ROW_W
    host_offsets = np.cumsum([0, *prompt_lens], dtype=np.uint32)
    per_row = run_per_row(
        per_row_host_step,
        prompt_lens,
        drafts,
        scores,
        coeffs,
        argmax,
        merged_len,
        host_offsets=host_offsets,
    )
    # The uniform narrow verify binds only the first w drafts per row.
    narrow = run_per_row(
        uniform_host_step,
        prompt_lens,
        drafts[:, :PER_ROW_W],
        scores,
        coeffs,
        argmax,
        merged_len,
        host_offsets=host_offsets,
    )
    assert len(per_row) == len(narrow) + 1
    for got, want in zip((*per_row[:3], *per_row[4:]), narrow, strict=True):
        np.testing.assert_array_equal(got, want)
    # Row 0 accepts exactly w: the mask stops the run at the verified
    # prefix, and a dropped or short mask would accept one more.
    assert per_row[0][0] == PER_ROW_W
    np.testing.assert_array_equal(per_row[3], [PER_ROW_W] * PER_ROW_B)


def _run_host_mirror(
    model: Model,
    prompt_lens: list[int],
    host_offsets: npt.NDArray[np.uint32],
    width: int,
) -> npt.NDArray[np.uint32]:
    """Returns the host offsets produced by one block step.

    Args:
        model: The compiled step with a host mirror.
        prompt_lens: The prompt lengths defining the device offsets.
        host_offsets: The row boundaries supplied on the host.
        width: The verify slots per row in the packed extent.

    Returns:
        The merged host row offsets.
    """
    batch = len(prompt_lens)
    *_, mirror = run_per_row(
        model,
        prompt_lens,
        np.zeros((batch, PER_ROW_K), dtype=np.int64),
        np.zeros((batch, PER_ROW_K), dtype=np.float32),
        np.array([1.0] + [0.0] * PER_ROW_K, dtype=np.float32),
        np.zeros(batch, dtype=np.int64),
        sum(prompt_lens) + batch * width,
        host_offsets=host_offsets,
    )
    return mirror


@pytest.mark.parametrize(
    ("prompt_lens", "widths"),
    [([3, 4], (1, PER_ROW_W, PER_ROW_K)), ([1, 1, 1], (PER_ROW_W,))],
)
def test_the_host_mirror_strides_by_the_step_width(
    prompt_lens: list[int],
    widths: tuple[int, ...],
    per_row_host_step: Model,
    uniform_host_step: Model,
) -> None:
    """The driver's host mirror strides by w under per-row and by the
    drafts' own width on the uniform path: only the replica boundaries
    are read, and each replica's rows sum to B_r * w."""
    offsets = np.cumsum([0, *prompt_lens], dtype=np.uint32)
    rows = np.arange(offsets.size, dtype=np.uint32)
    for width in widths:
        mirror = _run_host_mirror(
            per_row_host_step, prompt_lens, offsets, width
        )
        np.testing.assert_array_equal(mirror, offsets + rows * width)

    uniform = _run_host_mirror(
        uniform_host_step, prompt_lens, offsets, PER_ROW_K
    )
    np.testing.assert_array_equal(uniform, offsets + rows * PER_ROW_K)


def test_host_mirror_uses_host_row_boundaries(per_row_host_step: Model) -> None:
    host_offsets = np.array([0, 2, 7], dtype=np.uint32)
    mirror = _run_host_mirror(per_row_host_step, [3, 4], host_offsets, width=2)
    np.testing.assert_array_equal(mirror, [0, 4, 11])


def test_host_mirror_with_scalar_width(session: InferenceSession) -> None:
    cpu = DeviceRef.CPU()
    with Graph(
        "host_mirror",
        input_types=[
            TensorType(DType.uint32, ["input_row_offsets_len"], cpu),
            TensorType(DType.uint32, [], cpu),
        ],
    ) as graph:
        offsets_in, width_in = (v.tensor for v in graph.inputs)
        graph.output(compute_host_merged_offsets(offsets_in, width_in))
    model = session.load(graph)
    offsets = np.array([0, 3, 7], dtype=np.uint32)
    for width in (1, 2, 4):
        (out,) = model.execute(
            Buffer.from_numpy(offsets),
            Buffer.from_numpy(np.array(width, dtype=np.uint32)),
        )
        assert isinstance(out, Buffer)
        np.testing.assert_array_equal(
            out.to_numpy(), offsets + np.arange(3, dtype=np.uint32) * width
        )
