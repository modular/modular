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
"""Top-level pipeline dispatch helpers for cascade.

Selects and builds the correct :class:`CascadePipeline` for a
:class:`PipelineConfig` by resolving the model's architecture against the MAX
:obj:`~max.pipelines.PIPELINE_REGISTRY` and building the cascade pipeline class
the architecture declares
(:attr:`~max.pipelines.lib.registry.SupportedArchitecture.cascade_pipeline_factory`).
There is no task-based routing: the resulting pipeline's interfaces (see
``serve.all_routes``) determine which HTTP routes it serves.

The ``dummy_textgen`` and ``dummy_imgen`` model paths are in-process test
fixtures with no Hugging Face config, so they are selected by an exact
model-path match and never touch the registry.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from max.experimental.cascade.interfaces.pipeline import CascadePipeline
from max.experimental.cascade.pipelines.dummy_imgen import (
    build_dummy_imgen_pipeline,
)
from max.experimental.cascade.pipelines.dummy_textgen import (
    build_dummy_textgen_pipeline,
)
from max.pipelines.architectures import register_all_models
from max.pipelines.lib import PIPELINE_REGISTRY
from max.pipelines.lib.config import PipelineConfig
from max.pipelines.lib.registry import SupportedArchitecture

# Dummy pipelines are in-process test fixtures selected by an exact model-path
# sentinel; they have no Hugging Face config and never hit the registry.
_DUMMY_BUILDERS: dict[str, Callable[[], Awaitable[CascadePipeline]]] = {
    "dummy_textgen": build_dummy_textgen_pipeline,
    "dummy_imgen": build_dummy_imgen_pipeline,
}


def count_unique_device_specs(config: PipelineConfig) -> int:
    """Count the unique GPU ``device_specs`` sets across pipeline components.

    Each distinct set of GPU device IDs is treated as needing its own worker
    process; components sharing the same GPUs share a process. Used by the
    serve CLI to size the ``gpu`` worker pool from the model configuration.

    Args:
        config: The pipeline configuration whose components to inspect.

    Returns:
        The number of unique GPU ``device_specs`` sets, at least 1.
    """
    unique: set[frozenset[tuple[str, int]]] = set()
    for component in config.models.values():
        gpu_specs = frozenset(
            (spec.device_type, spec.id)
            for spec in component.device_specs
            if spec.device_type == "gpu"
        )
        if gpu_specs:
            unique.add(gpu_specs)
    return max(1, len(unique))


def _resolve_architecture(config: PipelineConfig) -> SupportedArchitecture:
    """Resolve the ``SupportedArchitecture`` for *config*'s main model.

    Registers the built-in architectures (idempotent), reads the architecture
    class name from the model's Hugging Face config, and looks it up in the
    MAX registry.

    Args:
        config: Fully-specified ``PipelineConfig`` for a real (non-dummy) model.

    Raises:
        ValueError: If the architecture is unknown to the MAX registry.
    """
    register_all_models()
    architecture_name = config.models.main_architecture_name
    arch = PIPELINE_REGISTRY.retrieve_architecture(
        architecture_name=architecture_name,
        prefer_module_v3=config.runtime.prefer_module_v3,
    )
    if arch is None:
        raise ValueError(
            f"No MAX architecture found for {architecture_name!r}."
        )
    return arch


async def build_pipeline(
    config: PipelineConfig,
) -> CascadePipeline:
    """Build the cascade pipeline described by *config*.

    Dummy fixtures (``dummy_textgen`` / ``dummy_imgen``) are matched by exact
    model path. Every other model is resolved against the MAX architectures
    registry, and its
    :attr:`~max.pipelines.lib.registry.SupportedArchitecture.cascade_pipeline_factory`
    class is constructed from *config*.

    Args:
        config: Fully-specified ``PipelineConfig``, typically constructed
            by cyclopts from ``--models.*`` CLI flags.

    Raises:
        ValueError: If no model is specified.
        NotImplementedError: If the architecture declares no cascade pipeline.
    """
    if not config.models:
        raise ValueError(
            "No models specified. Pass a model path via "
            "--models.main.model-path <repo-id>."
        )

    model_path = config.model.model_path
    if dummy_builder := _DUMMY_BUILDERS.get(model_path):
        return await dummy_builder()

    arch = _resolve_architecture(config)
    factory = arch.cascade_pipeline_factory
    if factory is None:
        raise NotImplementedError(
            f"Architecture {arch.name!r} (model {model_path!r}) has no cascade "
            "pipeline. Set cascade_pipeline_factory on its "
            "SupportedArchitecture to enable cascade serving."
        )

    pipeline = factory(config)
    assert isinstance(pipeline, CascadePipeline)
    return pipeline
