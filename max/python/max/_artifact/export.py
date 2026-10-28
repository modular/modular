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
"""Exports a pipeline to a ``.m`` pipeline artifact."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from max.graph import SymbolicDim
from max.pipelines.lib.config import PipelineConfig
from max.pipelines.lib.graph_specs import GraphRole, compile_roles_separately
from max.pipelines.lib.interfaces import PipelineModelWithKVCache
from max.pipelines.lib.memory_estimation import MemoryPlan
from max.pipelines.lib.pipeline_variants.text_generation import (
    TextGenerationPipelineInterface,
)
from max.pipelines.lib.registry import PIPELINE_REGISTRY
from max.tensor.compilation import CompiledCallable
from max.tensor.sharding import BufferLayout, TensorLayout

from .batch_processor import BatchProcessorDims
from .engine_config import (
    EngineConfig,
    EngineSamplingConfig,
    EngineSpeculativeConfig,
)
from .model_artifact import ModelArtifact
from .pipeline_modules import PipelineModules
from .serialization import serialize
from .value_types import ARTIFACT_SUFFIX, ArtifactError

logger = logging.getLogger("max.pipelines")


@runtime_checkable
class SupportsCompiledModules(Protocol):
    """A pipeline that keeps each role's compiled graph.

    The text-generation pipelines work this way.
    """

    @property
    def compiled_modules(self) -> Mapping[str, CompiledCallable[..., Any]]:
        """Each role's compiled graph, keyed by role."""
        ...


def pipeline_modules(
    pipeline: SupportsCompiledModules, architecture: str
) -> PipelineModules:
    """Describes a built pipeline's compiled graphs, keyed by role.

    The graphs are the pipeline's ``compiled_modules``: a text-generation
    pipeline's model's ``graph_specs`` roles and its samplers.

    Args:
        pipeline: The built pipeline.
        architecture: The pipeline's architecture, named in errors.

    Returns:
        The graphs, ready to store as the artifact's ``model`` component.

    Raises:
        ArtifactError: If the pipeline registers no ``decoder`` or
            ``spec_decoder`` graph, which is the case until its pipeline
            model implements ``graph_specs``.
    """
    modules = pipeline.compiled_modules
    if not (GraphRole.DECODER in modules or GraphRole.SPEC_DECODER in modules):
        raise ArtifactError(
            f"{architecture} registers no decoder graph by role; its pipeline"
            " model must implement graph_specs before it can be exported"
        )
    # A GraphRole formats as its member name, so key by its value.
    return PipelineModules(
        {
            role.value if isinstance(role, GraphRole) else role: module
            for role, module in modules.items()
        }
    )


def batch_processor_dims(
    model: PipelineModelWithKVCache[Any], decoder: CompiledCallable[..., Any]
) -> BatchProcessorDims | None:
    """Bounds each symbolic dim of the decoder's per-step inputs.

    Buffers are the inputs an owner allocates once for the model's lifetime,
    such as KV pages or a state pool, and sizes itself. Every other input is
    staged each step, so the serving engine needs the largest size of its
    dims, which :func:`per_step_dims` lists.

    Args:
        model: The built pipeline model.
        decoder: Its compiled decoder.

    Returns:
        The bounds, or ``None`` when the model has not named its graph
        inputs to its batch processor.
    """
    processor = model.batch_processor
    if processor is None or processor.input_tree is None:
        return None
    return BatchProcessorDims(
        processor.runtime.max_dims(
            per_step_dims(decoder.in_layouts), model.kv_params
        )
    )


def per_step_dims(layouts: Sequence[TensorLayout]) -> list[str]:
    """Returns each symbolic dim of a non-buffer input, in first use."""
    return list(
        {
            dim.name: None
            for layout in layouts
            if not isinstance(layout, BufferLayout)
            for dim in layout.shape
            if isinstance(dim, SymbolicDim)
        }
    )


def pipeline_engine_config(
    pipeline_config: PipelineConfig, memory_plan: MemoryPlan
) -> EngineConfig:
    """Returns the settings a pipeline is built with, for its artifact.

    The batch size, maximum length, batch token cap and devices come from the
    memory plan the pipeline is built with: the configuration holds only what
    was asked for, which planning resolves or lowers.

    Args:
        pipeline_config: The pipeline's configuration.
        memory_plan: The memory plan the pipeline is built with.

    Returns:
        The artifact's ``engine_config`` component.

    Raises:
        ArtifactError: If the plan has no maximum length, as for a pipeline
            with no language model.
    """
    max_length = memory_plan.planned_max_length
    if max_length is None:
        raise ArtifactError("the memory plan sets no maximum length")
    runtime = pipeline_config.runtime
    sampling = pipeline_config.sampling
    speculative = pipeline_config.speculative
    spec_config: EngineSpeculativeConfig | None = None
    if speculative is not None:
        # The config keeps a speculative section only when a method is set.
        assert speculative.speculative_method is not None
        spec_config = EngineSpeculativeConfig(
            speculative_method=speculative.speculative_method,
            num_speculative_tokens=speculative.draft_width,
            draft_proposal=speculative.draft_proposal,
            reasoning_parser=runtime.reasoning_parser,
        )
    return EngineConfig(
        max_batch_size=memory_plan.planned_max_batch_size,
        max_length=max_length,
        max_batch_input_tokens=runtime.max_batch_input_tokens,
        max_batch_total_tokens=memory_plan.planned_max_batch_total_tokens,
        devices=memory_plan.require_device_specs(),
        data_parallel_degree=pipeline_config.model.data_parallel_degree,
        ep_size=runtime.ep_size,
        sampling=EngineSamplingConfig(
            needs_bitmask_constraints=pipeline_config.needs_bitmask_constraints,
            enable_variable_logits=sampling.enable_variable_logits,
            enable_penalties=sampling.enable_penalties,
        ),
        speculative=spec_config,
    )


def export_pipeline(
    pipeline_config: PipelineConfig, path: str | os.PathLike[str]
) -> Path:
    """Builds the pipeline for a config and writes it as a ``.m`` artifact.

    Builds the pipeline the same way serving does, so the artifact holds the
    graphs a served pipeline compiles and the settings it is built with.

    Args:
        pipeline_config: The pipeline's configuration.
        path: Where to write the artifact. ``.m`` is appended unless present.

    Returns:
        The path of the written artifact.

    Raises:
        ArtifactError: If the pipeline cannot be exported, or the destination
            exists.
    """
    dest = Path(path)
    if dest.suffix != ARTIFACT_SUFFIX:
        dest = dest.with_name(dest.name + ARTIFACT_SUFFIX)
    # Fail before the build, which can take minutes; serialize checks again.
    if dest.exists():
        raise ArtifactError(f"artifact destination {dest} already exists")
    architecture = pipeline_config.models.main_architecture_name
    task = PIPELINE_REGISTRY.retrieve_pipeline_task(architecture)
    with compile_roles_separately():
        retrieved = PIPELINE_REGISTRY.retrieve_factory(
            pipeline_config, task=task
        )
        pipeline = retrieved.factory()
    if not isinstance(pipeline, SupportsCompiledModules):
        raise ArtifactError(
            f"{architecture} builds a {type(pipeline).__name__}, which does"
            " not record its compiled graphs by role"
        )
    model = pipeline_modules(pipeline, architecture)
    dims = (
        batch_processor_dims(pipeline._pipeline_model, model.roles["decoder"])
        if isinstance(pipeline, TextGenerationPipelineInterface)
        and "decoder" in model.roles
        else None
    )
    engine_config = pipeline_engine_config(
        pipeline_config, retrieved.memory_plan
    )
    written = serialize(
        ModelArtifact(
            model=model, engine_config=engine_config, batch_processor=dims
        ),
        dest,
    )
    logger.info("Exported graphs %s to %s", sorted(model.roles), written)
    return written
