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
"""Checks per-row draft-prefix merging with a symbolic packed extent."""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.pipelines.speculative.draft_prefix import merge_draft_prefixes
from ragged_prefix_checks import (
    MergePrefixes,
    check_decode_rows_with_zero_and_partial_lengths,
    check_empty_first_row_full_second_row,
    check_mixed_prefill_rows,
    check_seeded_random_lengths,
    check_single_row,
    check_uniform_lengths_match_existing_merger,
    merge_outputs,
    prefix_input_types,
)


@pytest.fixture(scope="module")
def prefix_merge() -> MergePrefixes:
    """Returns a merge runner whose packed extent is bound by an input."""
    device = DeviceRef.CPU()
    with Graph(
        "merge_draft_prefixes",
        input_types=(
            *prefix_input_types(device),
            TensorType(DType.uint8, ["packed_length"], device=device),
        ),
    ) as graph:
        prompt, offsets, drafts, lengths, packed = (
            v.tensor for v in graph.inputs
        )
        graph.output(
            *merge_draft_prefixes(
                prompt, offsets, drafts, lengths, packed.shape[0]
            )
        )
    model = InferenceSession(devices=[CPU()]).load(graph)

    def merge(
        prompt: np.ndarray,
        offsets: np.ndarray,
        drafts: np.ndarray,
        lengths: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        packed = np.empty(int(lengths.sum()), dtype=np.uint8)
        return merge_outputs(model, prompt, offsets, drafts, lengths, packed)

    return merge


def test_decode_rows_with_zero_and_partial_lengths(
    prefix_merge: MergePrefixes,
) -> None:
    check_decode_rows_with_zero_and_partial_lengths(prefix_merge)


def test_mixed_prefill_rows(prefix_merge: MergePrefixes) -> None:
    check_mixed_prefill_rows(prefix_merge)


def test_uniform_lengths_match_existing_merger(
    prefix_merge: MergePrefixes,
) -> None:
    check_uniform_lengths_match_existing_merger(prefix_merge)


def test_single_row(prefix_merge: MergePrefixes) -> None:
    check_single_row(prefix_merge)


def test_empty_first_row_full_second_row(prefix_merge: MergePrefixes) -> None:
    check_empty_first_row_full_second_row(prefix_merge)


def test_seeded_random_lengths(prefix_merge: MergePrefixes) -> None:
    check_seeded_random_lengths(prefix_merge)
