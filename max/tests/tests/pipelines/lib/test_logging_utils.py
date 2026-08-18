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
"""Tests for log_basic_config output.

Guards against a regression where max_seq_len was missing from server startup
logs because log_basic_config was called before pipeline_config.resolve() had
populated model.max_length. (cache_memory now logs from the memory planner,
where the KV budget is computed, rather than from this config logger.)
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

from max.driver import DeviceSpec
from max.pipelines.lib import (
    KVCacheConfig,
    MAXModelConfig,
    PipelineConfig,
    PipelineRuntimeConfig,
)
from max.pipelines.lib.model_manifest import ModelManifest
from max.pipelines.logging_utils import log_basic_config
from max.pipelines.modeling.types import PipelineTask


def _make_pipeline_config(max_length: int | None) -> PipelineConfig:
    """Build a minimal PipelineConfig without triggering full validation."""
    model_config = MAXModelConfig.model_construct(
        model_path="modularai/Llama-3.1-8B-Instruct-GGUF",
        device_specs=[DeviceSpec.cpu()],
        max_length=max_length,
    )
    model_config.kv_cache = KVCacheConfig()
    model_config._huggingface_config = MagicMock()

    runtime = PipelineRuntimeConfig.model_construct()
    return PipelineConfig.model_construct(
        runtime=runtime,
        models=ModelManifest({"main": model_config}),
    )


def _capture_log_basic_config(config: PipelineConfig) -> str:
    """Call log_basic_config with mocked registry and return all logged lines."""
    arch = MagicMock()
    arch.name = "LlamaForCausalLM"
    arch.task = PipelineTask.TEXT_GENERATION

    pipeline_cls = type("TextGenerationPipeline", (), {})

    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    capture = _Capture()
    logger = logging.getLogger("max.pipelines")
    original_level = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(capture)
    try:
        with (
            patch(
                "max.pipelines.logging_utils.PIPELINE_REGISTRY.retrieve_architecture",
                return_value=arch,
            ),
            patch(
                "max.pipelines.logging_utils.get_pipeline_for_task",
                return_value=pipeline_cls,
            ),
        ):
            log_basic_config(config)
    finally:
        logger.removeHandler(capture)
        logger.setLevel(original_level)

    return "\n".join(r.getMessage() for r in records)


class TestLogBasicConfigAfterResolve:
    """log_basic_config must reflect resolved config values."""

    def test_max_seq_len_present_after_resolve(self) -> None:
        """max_seq_len must show the resolved value, not None."""
        config = _make_pipeline_config(max_length=131072)
        output = _capture_log_basic_config(config)
        assert "131072" in output, (
            f"Expected max_seq_len=131072 in log output:\n{output}"
        )
