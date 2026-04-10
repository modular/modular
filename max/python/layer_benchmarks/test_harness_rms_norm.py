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

"""Harness tests for RMSNorm (benchmark + correctness)."""

from __future__ import annotations

import pytest
from testbed.harnesses.rms_norm import (
    RMSNormDynamicParams,
    RMSNormHarness,
    RMSNormStaticParams,
)
from testbed.runner import LayerTestRunner, create_session

_STATIC_PARAMS = RMSNormStaticParams(dim=4096, eps=1e-6)

_SMOKE_SHAPES = [
    RMSNormDynamicParams(batch_size=1, seq_len=1024),
    RMSNormDynamicParams(batch_size=1, seq_len=1),
]


@pytest.fixture(scope="module")
def runner() -> LayerTestRunner[
    RMSNormStaticParams, RMSNormDynamicParams, None
]:
    session, device = create_session()
    return LayerTestRunner(RMSNormHarness(_STATIC_PARAMS, session, device))


def test_benchmark_smoke(
    runner: LayerTestRunner[RMSNormStaticParams, RMSNormDynamicParams, None],
) -> None:
    results = runner.benchmark(_SMOKE_SHAPES, iterations=1, warmup=1)
    for _label, stats in results:
        assert stats.mean_ms > 0.0


def test_correctness(
    runner: LayerTestRunner[RMSNormStaticParams, RMSNormDynamicParams, None],
) -> None:
    results = runner.correctness(
        _SMOKE_SHAPES, atol=1e-2, rtol=1e-2, cos_threshold=0.001
    )
    for r in results:
        assert r.passed, f"Correctness failed for {r.label}: {r}"
