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
"""The grouped ``wo_a`` projection's group-rank requirements, per arm.

The 128-row interleave constraint belongs to the block-diagonal launch
alone (its B-scales are one interleave over the whole shard, viewed per
group); the single-group dense path and the per-group reference arm need
only whole 32-row scale blocks, so a rank that is a multiple of 32 but
not of 128 -- the reduced fixtures and smaller V4.1-family configs the
bf16 baseline served -- must keep tracing. CPU-only: graph construction
picks the reference arm here, which is exactly the arm a 64-row group
falls back to on SM100 too.
"""

from __future__ import annotations

import pytest
from max import functional as F
from max.dtype import DType
from max.pipelines.architectures.deepseekV41_modulev3.layers.mxfp8_linear import (
    grouped_dense_op,
)
from max.tensor import Tensor


def _stage(group_rows: int, groups: int, rows: int) -> None:
    """Stages ``grouped_dense_op`` over an E4M3/E8M0 leaf, without raising."""
    k = 64
    with F.lazy():
        x = Tensor.zeros([rows, groups * k], dtype=DType.bfloat16)
        weight = Tensor.zeros(
            [rows, k], dtype=DType.float8_e4m3fn, device=x.device
        )
        scale = Tensor.zeros(
            [rows // 32, k // 32], dtype=DType.float8_e8m0fnu, device=x.device
        )
        out = grouped_dense_op(x, weight, scale, group_rows)
        assert out.dtype == DType.bfloat16


@pytest.mark.parametrize(
    ("group_rows", "groups"),
    [(64, 2), (96, 1), (128, 2), (1024, 1)],
)
def test_group_ranks_that_are_whole_32_row_blocks_trace(
    group_rows: int, groups: int
) -> None:
    """A rank that is a whole 32-row scale block traces, 128 or not."""
    _stage(group_rows, groups, groups * group_rows)


def test_group_rank_below_a_scale_block_fails_loudly() -> None:
    """A rank that would split a 32-row scale block is refused.

    A rank of 16 leaves its first group with no scale rows of its own
    (16 // 32 = 0): the reference arm would slice its compact scales an
    empty range and die inside the matmul wrapper, and the single-group
    path would read a degenerate zero-row scale leaf as fabricated unit
    scales -- a loud, up-front refusal instead of either.
    """
    with pytest.raises(ValueError, match="32-row scale blocks"):
        _stage(16, 2, 32)


if __name__ == "__main__":
    raise SystemExit("run via bazel or pytest")
