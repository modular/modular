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
"""Shared fixtures for benchmark unit tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from max.benchmark import benchmark_serving
from max.benchmark.benchmark_shared.datasets import (
    BenchmarkDataset,
    SampledRequest,
    ShareGPTBenchmarkDataset,
)
from max.benchmark.benchmark_shared.datasets.types import RequestSamples


@pytest.fixture
def offline_dryrun_mocks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock tokenizer + dataset loaders so ``--dry-run`` tests don't need network.

    Opt-in via ``pytestmark = pytest.mark.usefixtures("offline_dryrun_mocks")``
    at module scope. The unified dry-run path always loads the tokenizer
    and builds the dataset before printing its summary; the bazel
    sandbox blocks network, so anything that exercises
    ``main_with_parsed_args`` needs this. Tests that exercise
    ``BenchmarkDataset.from_flags`` directly should *not* use this
    fixture.
    """
    mock_tokenizer = MagicMock(model_max_length=4096)
    monkeypatch.setattr(
        benchmark_serving, "get_tokenizer", lambda *a, **kw: mock_tokenizer
    )
    monkeypatch.setattr(
        benchmark_serving, "resolve_revision", lambda *a, **kw: None
    )

    mock_dataset = MagicMock(spec=ShareGPTBenchmarkDataset)
    mock_dataset.has_multiturn_chat_support = False
    mock_dataset.sample_requests.return_value = RequestSamples(
        requests=[
            SampledRequest(
                prompt_formatted="hi",
                prompt_len=2,
                output_len=8,
                encoded_images=[],
                ignore_eos=False,
            )
        ]
    )
    monkeypatch.setattr(
        BenchmarkDataset, "from_flags", MagicMock(return_value=mock_dataset)
    )
