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

"""Pipeline logging utilities.

Free functions for logging pipeline and architecture information at startup.
These live outside :class:`~max.pipelines.PipelineConfig` so that
``config.py`` stays free of module-level registry imports — config is pure
data; the registry provides the arch context needed for display.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from max.pipelines.lib.config.model_config import _format_config_entries
from max.pipelines.lib.registry import PIPELINE_REGISTRY, get_pipeline_for_task
from max.pipelines.modeling.types.task import PipelineTask
from max.support.human_readable_formatter import to_human_readable_bytes

if TYPE_CHECKING:
    from max.pipelines.lib.config.config import PipelineConfig

__all__ = ["log_basic_config", "log_pipeline_info"]

_logger = logging.getLogger("max.pipelines")


def log_pipeline_info(pipeline_config: PipelineConfig) -> None:
    """Logs comprehensive pipeline and KVCache configuration information.

    Args:
        pipeline_config: The resolved pipeline configuration to log.

    Raises:
        ValueError: If no architecture is found for the model. This should not
            happen after config resolution.
    """
    arch = PIPELINE_REGISTRY.retrieve_architecture(
        architecture_name=pipeline_config.models.main_architecture_name,
        prefer_module_v3=pipeline_config.runtime.prefer_module_v3,
    )

    if arch is None:
        raise ValueError(
            f"No architecture found for {pipeline_config.models.main_architecture_name}. "
            "This should not happen after config resolution."
        )
    pipeline_class = get_pipeline_for_task(arch.task, pipeline_config)

    arch_entries: list[tuple[str, Any]] = [
        ("architecture", arch.name),
        ("pipeline_class", pipeline_class.__name__),
        ("pipeline_model", arch.pipeline_model.__name__),
        ("tokenizer", arch.tokenizer_cls.__name__),
    ]

    _logger.info("")
    _logger.info("Pipeline Architecture")
    _logger.info("=" * 60)
    for line in _format_config_entries(arch_entries):
        _logger.info(line)

    pipeline_config.models.log_model_info()
    pipeline_entries: list[tuple[str, Any]] = []
    if "main" in pipeline_config.models:
        pipeline_entries.append(
            ("max_seq_len", pipeline_config.model.max_length)
        )
    pipeline_entries.extend(
        [
            ("chunked_prefill", pipeline_config.runtime.enable_chunked_prefill),
            (
                "max_batch_input_tokens",
                pipeline_config.runtime.max_batch_input_tokens,
            ),
            (
                "in_flight_batching",
                pipeline_config.runtime.enable_in_flight_batching,
            ),
        ]
    )

    _logger.info("")
    _logger.info("Pipeline Config")
    _logger.info("=" * 60)
    for line in _format_config_entries(pipeline_entries):
        _logger.info(line)
    _logger.info("")

    # Denoising cache details for diffusion pipelines.
    if arch.task == PipelineTask.PIXEL_GENERATION:
        cache = pipeline_config.runtime.denoising_cache
        cache_entries: list[tuple[str, Any]] = [
            ("first_block_caching", cache.first_block_caching),
            ("taylorseer", cache.taylorseer),
            (
                "taylorseer_cache_interval",
                cache.taylorseer_cache_interval
                if cache.taylorseer_cache_interval is not None
                else "model-default",
            ),
            (
                "taylorseer_warmup_steps",
                cache.taylorseer_warmup_steps
                if cache.taylorseer_warmup_steps is not None
                else "model-default",
            ),
            (
                "taylorseer_max_order",
                cache.taylorseer_max_order
                if cache.taylorseer_max_order is not None
                else "model-default",
            ),
        ]

        _logger.info("Denoising Cache")
        _logger.info("=" * 60)
        for line in _format_config_entries(cache_entries):
            _logger.info(line)
        _logger.info("")


def log_basic_config(pipeline_config: PipelineConfig) -> None:
    """Logs minimal pipeline configuration information.

    Logs basic pipeline options including model name, pipeline task,
    weight path, max_batch_size, max_seq_len, and reserved memory.

    Args:
        pipeline_config: The resolved pipeline configuration to log.

    Raises:
        ValueError: If no architecture is found for the model. This should not
            happen after config resolution.
    """
    arch = PIPELINE_REGISTRY.retrieve_architecture(
        architecture_name=pipeline_config.models.main_architecture_name,
        prefer_module_v3=pipeline_config.runtime.prefer_module_v3,
    )

    if arch is None:
        model_path = (
            pipeline_config.models.main_architecture_name
            if "main" in pipeline_config.models
            else str(list(pipeline_config.models.keys()))
        )
        raise ValueError(
            f"No architecture found for {model_path}. "
            "This should not happen after config resolution."
        )

    task = arch.task
    pipeline_class = get_pipeline_for_task(task, pipeline_config)

    kv_cache_tasks = {PipelineTask.TEXT_GENERATION}

    memory_str = None
    if "main" in pipeline_config.models and task in kv_cache_tasks:
        kv_config = pipeline_config.model.kv_cache
        if kv_config._available_cache_memory is not None:
            memory_str = to_human_readable_bytes(
                kv_config._available_cache_memory
            )

    config_entries: list[tuple[str, Any]] = [
        ("architecture", arch.name),
        ("pipeline", pipeline_class.__name__),
    ]
    if "main" in pipeline_config.models:
        devices_str = ", ".join(
            f"{d.device_type}[{d.id}]"
            for d in pipeline_config.model.device_specs
        )
        config_entries.extend(
            [
                ("model", pipeline_config.model.model_path),
                ("devices", devices_str),
                ("max_seq_len", pipeline_config.model.max_length),
            ]
        )

    if memory_str:
        config_entries.append(("cache_memory", memory_str))
    config_entries.append(
        ("device_graph_capture", pipeline_config.runtime.device_graph_capture)
    )

    if pipeline_config.speculative is not None:
        config_entries.append(
            (
                "speculative_method",
                pipeline_config.speculative.speculative_method,
            )
        )
        config_entries.append(
            (
                "num_speculative_tokens",
                pipeline_config.speculative.num_speculative_tokens,
            )
        )
        if pipeline_config.speculative.use_relaxed_acceptance_for_thinking:
            config_entries.append(
                ("relaxed_topk", pipeline_config.speculative.relaxed_topk)
            )
            config_entries.append(
                ("relaxed_delta", pipeline_config.speculative.relaxed_delta)
            )

    _logger.info("")
    _logger.info("=" * 60)
    _logger.info(
        "Pipeline Configuration (use --pretty-print-config to print full config)"
    )
    _logger.info("=" * 60)
    for line in _format_config_entries(config_entries):
        _logger.info(line)
    _logger.info("")
