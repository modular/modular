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

This is a temporary string-matching shim until cascade pipeline selection can
integrate with the MAX architectures registry. For now it only knows how to
build the in-process dummy pipelines used by the experimental CLI and tests.
"""

from __future__ import annotations

from max.experimental.cascade.interfaces.pipeline import CascadePipeline
from max.experimental.cascade.pipelines.common_textgen import (
    build_common_textgen_pipeline,
)
from max.experimental.cascade.pipelines.dummy_imgen import (
    build_dummy_imgen_pipeline,
)
from max.experimental.cascade.pipelines.dummy_textgen import (
    build_dummy_textgen_pipeline,
)
from max.pipelines.lib.config import PipelineConfig


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


async def build_pipeline(
    config: PipelineConfig,
) -> CascadePipeline:
    """Build the cascade pipeline described by *config*.

    Routes to the correct builder based on the main ``model_path``.

    Args:
        config: Fully-specified ``PipelineConfig``, typically constructed
            by cyclopts from ``--models.*`` CLI flags.
    """
    model_path = config.model.model_path

    if model_path == "dummy_textgen":
        return await build_dummy_textgen_pipeline()

    if model_path == "dummy_imgen":
        return await build_dummy_imgen_pipeline()

    if "smollm" in model_path.lower():
        return await build_common_textgen_pipeline(config)

    raise ValueError(
        f"Unsupported model {model_path!r}. "
        "cascade currently supports the dummy_textgen and dummy_imgen "
        "pipelines, and SmolLM via the common text-generation pipeline."
    )
