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

"""Checks for CPU-testable helpers in ``kernels.py``.

``_fp6_format_code`` is the boundary between the Python ``"e2m3"``/``"e3m2"``
encoding names and the integer ``FP6_FORMAT`` op parameter; a silent
off-by-one or swapped mapping there would route a matmul to the wrong OCP
encoding without any shape or dtype check catching it.
"""

from __future__ import annotations

import pytest
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType
from max.nn.kernels import (
    _fp6_format_code,
    block_scaled_preshuffle_grouped_scale_4d,
)


@pytest.mark.parametrize(
    "fp6_format, expected_code",
    [
        ("e2m3", 0),
        ("e3m2", 1),
    ],
)
def test_fp6_format_code_matches_op_parameter(
    fp6_format: str, expected_code: int
) -> None:
    assert _fp6_format_code(fp6_format) == expected_code


def test_fp6_format_code_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="fp6_format must be one of"):
        _fp6_format_code("e1m4")


# The AMD preb path's per-step A-scale buffer holds one fixed-stride slot per
# expert. Sizing it by the batch's POST-expansion row count instead of one
# expert's is correct but roughly `top_k`-fold too large per slot, which is
# invisible at decode and fatal at a 6610-token Kimi K3 prefill: 896 x 105,760
# x 112 B is 10.6 GB on a model already at 305 of 309 GB. These pin that the
# bound is opt-in (so MiniMax M3 and Kimi K2.5 are untouched) and that the
# region the matmul addresses still fits inside what the bound allocates -- too
# small a bound is silent corruption, not a crash.
_K3_EXPERTS, _K3_TOP_K, _K3_ROW_BOUND, _K3_K_SCALES = 896, 16, 8192, 112
_M3_EXPERTS, _M3_TOP_K, _M3_ROW_BOUND, _M3_K_SCALES = 128, 4, 8192, 192


def _preshuffled_scale_rows(
    tokens: int,
    top_k: int,
    num_experts: int,
    k_scales: int,
    max_rows_per_expert: int | None,
) -> int:
    """Rows `block_scaled_preshuffle_grouped_scale_4d` allocates for a step."""
    with Graph(
        "preshuffle_scale_rows",
        input_types=[
            TensorType(
                DType.float8_e8m0fnu,
                (tokens * top_k, k_scales),
                device=DeviceRef.GPU(0),
            ),
            TensorType(
                DType.uint32, (num_experts + 1,), device=DeviceRef.GPU(0)
            ),
            TensorType(DType.uint32, (2,), device=DeviceRef.CPU()),
        ],
    ) as graph:
        a_scales, expert_start_indices, stats = graph.inputs
        out = block_scaled_preshuffle_grouped_scale_4d(
            a_scales.tensor,
            expert_start_indices.tensor,
            stats.tensor[0],
            stats.tensor[1],
            num_experts=num_experts,
            max_rows_per_expert=max_rows_per_expert,
        )
        return int(out.shape[0])


@pytest.mark.parametrize(
    "tokens, top_k, num_experts, k_scales",
    [
        (6610, _K3_TOP_K, _K3_EXPERTS, _K3_K_SCALES),
        (6610, _M3_TOP_K, _M3_EXPERTS, _M3_K_SCALES),
    ],
    ids=["kimi_k3", "minimax_m3"],
)
def test_preshuffled_scale_buffer_defaults_to_the_post_expansion_count(
    tokens: int, top_k: int, num_experts: int, k_scales: int
) -> None:
    """Omitting the bound must reproduce the old shape byte for byte."""
    assert (
        _preshuffled_scale_rows(tokens, top_k, num_experts, k_scales, None)
        == num_experts * tokens * top_k
    )


@pytest.mark.parametrize(
    "tokens, top_k, num_experts, k_scales, row_bound",
    [
        (6610, _K3_TOP_K, _K3_EXPERTS, _K3_K_SCALES, _K3_ROW_BOUND),
        (6610, _M3_TOP_K, _M3_EXPERTS, _M3_K_SCALES, _M3_ROW_BOUND),
        # Decode: the static budget over-bounds a 32-token step by design, and
        # must still allocate rather than round to nothing.
        (32, _K3_TOP_K, _K3_EXPERTS, _K3_K_SCALES, _K3_ROW_BOUND),
        # A bound that is exactly the token count but not a multiple of 32:
        # without the padding, the buffer is 14 rows short per slot.
        (6610, _K3_TOP_K, _K3_EXPERTS, _K3_K_SCALES, 6610),
    ],
    ids=[
        "kimi_k3_prefill",
        "minimax_m3_prefill",
        "kimi_k3_decode",
        "kimi_k3_unaligned_bound",
    ],
)
def test_preshuffled_scale_buffer_contains_the_addressed_region(
    tokens: int, top_k: int, num_experts: int, k_scales: int, row_bound: int
) -> None:
    """The allocation is one padded bound per expert and covers every read.

    The matmul reads slot `e` at `e * align_up(max_num_tokens_per_expert, 32)`,
    and `max_num_tokens_per_expert` is `min(tokens, row_bound)` -- one expert
    cannot hold more rows than the batch has tokens, since a token's `top_k`
    picks are distinct. The allocation has to cover the last slot's last row.
    """
    rows = _preshuffled_scale_rows(
        tokens, top_k, num_experts, k_scales, row_bound
    )
    assert rows == num_experts * ((row_bound + 31) // 32) * 32
    max_padded_m = ((min(tokens, row_bound) + 31) // 32) * 32
    assert rows >= num_experts * max_padded_m


@pytest.mark.parametrize("row_bound", [0, -64])
def test_preshuffled_scale_buffer_rejects_a_non_positive_bound(
    row_bound: int,
) -> None:
    """A bound that sizes slots to nothing must not reach the kernel."""
    with pytest.raises(ValueError, match="max_rows_per_expert must be > 0"):
        _preshuffled_scale_rows(
            32, _K3_TOP_K, _K3_EXPERTS, _K3_K_SCALES, row_bound
        )
