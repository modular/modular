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
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from max.experimental.compilation import CompiledCallable
from max.pipelines.lib.config import PipelineConfig
from max.pipelines.lib.graph_specs import GraphRole, compile_roles_separately
from max.pipelines.lib.registry import PIPELINE_REGISTRY

from .model_artifact import ModelArtifact
from .pipeline_modules import PipelineModules
from .serialization import serialize
from .value_types import ARTIFACT_SUFFIX, ArtifactError

logger = logging.getLogger("max.pipelines")


@runtime_checkable
class SupportsCompiledModules(Protocol):
    """A pipeline that keeps each role's compiled graph, as the
    text-generation pipelines do.
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


def export_pipeline(
    pipeline_config: PipelineConfig, path: str | os.PathLike[str]
) -> Path:
    """Builds the pipeline for a config and writes it as a ``.m`` artifact.

    Builds the pipeline the same way serving does, so the artifact holds the
    graphs a served pipeline compiles.

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
        _, pipeline = PIPELINE_REGISTRY.retrieve(pipeline_config, task=task)
    if not isinstance(pipeline, SupportsCompiledModules):
        raise ArtifactError(
            f"{architecture} builds a {type(pipeline).__name__}, which does"
            " not record its compiled graphs by role"
        )
    model = pipeline_modules(pipeline, architecture)
    written = serialize(ModelArtifact(model=model), dest)
    logger.info("Exported graphs %s to %s", sorted(model.roles), written)
    return written
