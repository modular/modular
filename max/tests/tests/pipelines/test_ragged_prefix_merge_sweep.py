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
"""Checks per-row draft-prefix merging with a static packed extent."""

from __future__ import annotations

from collections.abc import Callable
from functools import cache

import numpy as np
import pytest
from cpu_step_session import cpu_session as cpu_session
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Dim, Graph
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
def static_merge(session: InferenceSession) -> MergePrefixes:
    """Returns a merge runner compiled once per static packed extent."""

    @cache
    def load(total: int) -> Model:
        with Graph(
            "merge_draft_prefixes",
            input_types=prefix_input_types(DeviceRef.CPU()),
        ) as graph:
            prompt, offsets, drafts, lengths = (v.tensor for v in graph.inputs)
            graph.output(
                *merge_draft_prefixes(
                    prompt, offsets, drafts, lengths, Dim(total)
                )
            )
        return session.load(graph)

    def merge(
        prompt: np.ndarray,
        offsets: np.ndarray,
        drafts: np.ndarray,
        lengths: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return merge_outputs(
            load(int(lengths.sum())), prompt, offsets, drafts, lengths
        )

    return merge


@pytest.mark.parametrize(
    "check",
    [
        check_decode_rows_with_zero_and_partial_lengths,
        check_mixed_prefill_rows,
        check_uniform_lengths_match_existing_merger,
        check_single_row,
        check_empty_first_row_full_second_row,
        check_seeded_random_lengths,
    ],
)
def test_static_packed_extent(
    check: Callable[[MergePrefixes], None], static_merge: MergePrefixes
) -> None:
    check(static_merge)
