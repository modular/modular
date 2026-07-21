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
"""Tests for the PipelineArgs <-> PipelineConfig round-trip.

The serve entrypoint parses CLI flags into a ``PipelineConfig``, converts it
to ``PipelineArgs`` (``from_pipeline_config``), and later reconstructs the
config for the model worker (``PipelineConfig.from_args``). Any runtime field
missing from that round-trip is silently reset to its default in the worker,
so CLI flags appear accepted but never take effect.
"""

from __future__ import annotations

from max.pipelines.lib import PipelineArgs, PipelineConfig
from max.pipelines.lib.pipeline_runtime_config import PipelineRuntimeConfig


def test_from_args_threads_fold_sampler_and_pending_futures() -> None:
    args = PipelineArgs(fold_sampler_into_graph=True, max_pending_futures=2)
    config = PipelineConfig.from_args(args)
    assert config.runtime.fold_sampler_into_graph is True
    assert config.runtime.max_pending_futures == 2


def test_runtime_flags_survive_args_round_trip() -> None:
    # Non-default values for the fields this test guards.
    runtime = PipelineRuntimeConfig(
        fold_sampler_into_graph=True, max_pending_futures=2
    )
    config = PipelineConfig.from_args(PipelineArgs())
    config.runtime = runtime

    round_tripped = PipelineConfig.from_args(
        PipelineArgs.from_pipeline_config(config)
    )
    assert round_tripped.runtime.fold_sampler_into_graph is True
    assert round_tripped.runtime.max_pending_futures == 2
