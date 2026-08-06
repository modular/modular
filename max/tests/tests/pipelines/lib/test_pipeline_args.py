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
"""Tests for the flat-kwargs -> PipelineArgs -> PipelineConfig path.

The serve entrypoint parses CLI flags into ``PipelineArgs``
(``from_flat_kwargs``) and later constructs the config for the model worker
(``PipelineConfig.from_args``). Any runtime field dropped along that path is
silently reset to its default in the worker, so CLI flags appear accepted but
never take effect.
"""

from __future__ import annotations

from max.pipelines.lib import PipelineArgs, PipelineConfig
from max.pipelines.lib.model_manifest import ModelManifest
from max.pipelines.lib.pipeline_runtime_config import PipelineRuntimeConfig


def test_from_args_threads_fold_sampler_and_pending_futures() -> None:
    args = PipelineArgs(
        runtime=PipelineRuntimeConfig(
            fold_sampler_into_graph=True, max_pending_futures=2
        )
    )
    config = PipelineConfig.from_args(args)
    assert config.runtime.fold_sampler_into_graph is True
    assert config.runtime.max_pending_futures == 2


def test_runtime_flags_survive_flat_kwargs_path() -> None:
    # Non-default values for the fields this test guards, spelled the way
    # the CLI passes them (flat).
    config = PipelineConfig.from_args(
        PipelineArgs.from_flat_kwargs(
            fold_sampler_into_graph=True, max_pending_futures=2
        )
    )
    assert config.runtime.fold_sampler_into_graph is True
    assert config.runtime.max_pending_futures == 2


def test_empty_models_kwarg_is_not_a_manifest_override() -> None:
    # The CLI generates a --models flag from PipelineConfig's models field,
    # so every invocation carries an empty-manifest default. It must not be
    # taken as an explicit manifest override, or every CLI serve/generate
    # produces an empty-manifest config (regression: smoke tests failed at
    # startup with "Cannot determine architecture name: manifest is empty").
    args = PipelineArgs.from_flat_kwargs(models={}, max_batch_size=2)
    assert args._manifest_override is None
    assert PipelineArgs(models=ModelManifest())._manifest_override is None
