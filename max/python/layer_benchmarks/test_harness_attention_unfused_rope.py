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

"""Correctness test for the unfused rope+store path.

Verifies that ``_fuse_rope_and_store=False`` produces the same output as
the fused path by comparing both against the HuggingFace torch reference.
"""

from __future__ import annotations

import pytest
from max.interfaces import TextGenerationContext
from testbed.harnesses.attention_with_rope import (
    AttentionWithRopeHarness,
    AttentionWithRopeStaticParams,
)
from testbed.harnesses.ragged_attention_harness import AttentionDynamicParams
from testbed.runner import LayerTestRunner, create_session

_BASE_PARAMS = AttentionWithRopeStaticParams(
    hidden_size=4096,
    n_heads=32,
    n_kv_heads=8,
    head_dim=128,
    max_seq_len=65536,
    rope_theta=500000.0,
    dtype="bf16",
    _fuse_rope_and_store=False,
)

_CORRECTNESS_SHAPES = [
    AttentionDynamicParams(batch_size=1, seq_len=11),
    AttentionDynamicParams(batch_size=1, seq_len=128),
]


@pytest.fixture(scope="module")
def unfused_runner() -> LayerTestRunner[
    AttentionWithRopeStaticParams,
    AttentionDynamicParams,
    list[TextGenerationContext],
]:
    session, device = create_session()
    return LayerTestRunner(
        AttentionWithRopeHarness(_BASE_PARAMS, session, device)
    )


def test_unfused_correctness(
    unfused_runner: LayerTestRunner[
        AttentionWithRopeStaticParams,
        AttentionDynamicParams,
        list[TextGenerationContext],
    ],
) -> None:
    """Unfused rope+store path matches torch reference."""
    results = unfused_runner.correctness(
        _CORRECTNESS_SHAPES, atol=0.0625, rtol=0.016, cos_threshold=0.001
    )
    for r in results:
        assert r.passed, f"Correctness failed for {r.label}: {r}"
