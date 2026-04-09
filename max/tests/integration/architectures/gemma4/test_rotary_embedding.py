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
"""Tests for Gemma4 proportional rotary embedding."""

from __future__ import annotations

import pytest
import torch
from conftest import (  # type: ignore[import-not-found]
    TEXT_GLOBAL_HEAD_DIM,
    TEXT_GLOBAL_PARTIAL_ROTARY_FACTOR,
    TEXT_GLOBAL_ROPE_THETA,
    TEXT_HIDDEN_SIZE,
    TEXT_NUM_ATTENTION_HEADS,
    torch_compute_proportional_rope_inv_freqs,
)
from max.driver import CPU
from max.engine import InferenceSession
from max.graph import Graph
from max.pipelines.architectures.gemma4.layers.rotary_embedding import (
    ProportionalRotaryEmbedding,
    ProportionalScalingParams,
)

# Use a short max_seq_len to keep test execution fast.
_MAX_SEQ_LEN = 16


def _make_rope(
    head_dim: int = TEXT_GLOBAL_HEAD_DIM,
    theta: float = TEXT_GLOBAL_ROPE_THETA,
    partial_rotary_factor: float = TEXT_GLOBAL_PARTIAL_ROTARY_FACTOR,
    max_seq_len: int = _MAX_SEQ_LEN,
) -> ProportionalRotaryEmbedding:
    return ProportionalRotaryEmbedding(
        dim=TEXT_HIDDEN_SIZE,
        n_heads=TEXT_NUM_ATTENTION_HEADS,
        theta=theta,
        max_seq_len=max_seq_len,
        head_dim=head_dim,
        scaling_params=ProportionalScalingParams(
            partial_rotary_factor=partial_rotary_factor
        ),
    )


def _run_inv_freqs(rope: ProportionalRotaryEmbedding) -> torch.Tensor:
    """Execute a graph that returns the inverse frequencies tensor."""
    device = CPU()
    session = InferenceSession(devices=[device])
    with Graph("inv_freqs", input_types=[]) as graph:
        result = rope._compute_inv_freqs()
        graph.output(result)
    compiled = session.load(graph, weights_registry={})
    (buf,) = compiled.execute()
    return torch.from_dlpack(buf).cpu().float()


@pytest.mark.parametrize(
    ("head_dim", "partial_rotary_factor"),
    [
        (TEXT_GLOBAL_HEAD_DIM, TEXT_GLOBAL_PARTIAL_ROTARY_FACTOR),
        (TEXT_GLOBAL_HEAD_DIM, 0.5),
        (TEXT_GLOBAL_HEAD_DIM, 1.0),
        (256, TEXT_GLOBAL_PARTIAL_ROTARY_FACTOR),
    ],
    ids=[
        "gemma4_31b_global",
        "half_rotation",
        "full_rotation",
        "local_head_dim",
    ],
)
def test_inv_freqs_shape(head_dim: int, partial_rotary_factor: float) -> None:
    """Inverse frequencies must have shape [head_dim // 2]."""
    rope = _make_rope(
        head_dim=head_dim, partial_rotary_factor=partial_rotary_factor
    )
    result = _run_inv_freqs(rope)
    assert result.shape == (head_dim // 2,)


@pytest.mark.parametrize(
    ("head_dim", "partial_rotary_factor"),
    [
        (TEXT_GLOBAL_HEAD_DIM, TEXT_GLOBAL_PARTIAL_ROTARY_FACTOR),
        (TEXT_GLOBAL_HEAD_DIM, 0.5),
        (TEXT_GLOBAL_HEAD_DIM, 1.0),
        (256, TEXT_GLOBAL_PARTIAL_ROTARY_FACTOR),
    ],
    ids=[
        "gemma4_31b_global",
        "half_rotation",
        "full_rotation",
        "local_head_dim",
    ],
)
def test_inv_freqs_match_reference(
    head_dim: int, partial_rotary_factor: float
) -> None:
    """Inverse frequencies must match the HuggingFace reference."""
    rope = _make_rope(
        head_dim=head_dim, partial_rotary_factor=partial_rotary_factor
    )
    max_result = _run_inv_freqs(rope)
    ref = torch_compute_proportional_rope_inv_freqs(
        head_dim=head_dim,
        theta=TEXT_GLOBAL_ROPE_THETA,
        partial_rotary_factor=partial_rotary_factor,
    )
    torch.testing.assert_close(max_result, ref, rtol=1e-5, atol=1e-5)


def test_nope_suffix_is_zero() -> None:
    """Dimensions beyond the rotated fraction must have zero frequency."""
    rope = _make_rope()
    result = _run_inv_freqs(rope)
    rope_angles = int(
        TEXT_GLOBAL_PARTIAL_ROTARY_FACTOR * TEXT_GLOBAL_HEAD_DIM // 2
    )
    nope_angles = TEXT_GLOBAL_HEAD_DIM // 2 - rope_angles
    assert nope_angles > 0, "test requires nope_angles > 0"
    torch.testing.assert_close(
        result[rope_angles:],
        torch.zeros(nope_angles),
        rtol=0,
        atol=0,
    )


def test_rotated_prefix_is_nonzero() -> None:
    """Dimensions within the rotated fraction must have non-zero frequency."""
    rope = _make_rope()
    result = _run_inv_freqs(rope)
    rope_angles = int(
        TEXT_GLOBAL_PARTIAL_ROTARY_FACTOR * TEXT_GLOBAL_HEAD_DIM // 2
    )
    assert (result[:rope_angles] > 0).all()
