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

"""Harness tests for Gemma4Attention — benchmark only."""

from __future__ import annotations

import pytest
from max.interfaces import TextGenerationContext
from testbed.harnesses.gemma4_attention import (
    Gemma4AttentionHarness,
    Gemma4AttentionStaticParams,
)
from testbed.harnesses.ragged_attention_harness import AttentionDynamicParams
from testbed.runner import LayerTestRunner, create_session

# gemma4-12b sliding window config (reduced max_seq_len for test GPU memory)
_STATIC_PARAMS = Gemma4AttentionStaticParams(
    hidden_size=3840,
    n_heads=16,
    n_kv_heads=8,
    n_global_kv_heads=4,
    head_dim=256,
    global_head_dim=512,
    max_seq_len=16384,
    rope_theta=1000000.0,
    is_sliding=True,
    attention_k_eq_v=True,
    local_window_size=1024,
    total_num_pages=2048,
)

_SMOKE_SHAPES = [
    AttentionDynamicParams(batch_size=1, seq_len=1024, ctx_len=1024),
    AttentionDynamicParams(batch_size=1, seq_len=1, ctx_len=1024),
]


@pytest.fixture(scope="module")
def runner() -> LayerTestRunner[
    Gemma4AttentionStaticParams,
    AttentionDynamicParams,
    list[TextGenerationContext],
]:
    session, device = create_session()
    return LayerTestRunner(
        Gemma4AttentionHarness(_STATIC_PARAMS, session, device)
    )


def test_benchmark_smoke(
    runner: LayerTestRunner[
        Gemma4AttentionStaticParams,
        AttentionDynamicParams,
        list[TextGenerationContext],
    ],
) -> None:
    results = runner.benchmark(_SMOKE_SHAPES, iterations=1, warmup=1)
    for _label, stats in results:
        assert stats.mean_ms > 0.0
