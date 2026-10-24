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

"""Where a decoder may be cut into pipeline stages, and what each stage holds.

A stage holds a contiguous range of decoder layers. An exporter builds each
stage's graph from a config that carries its stage, and the engine that
serves the stages runs them on their own devices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ..pipeline_stage import PipelineStage

if TYPE_CHECKING:
    from transformers import AutoConfig

    from ..arch_lookup import SupportedArchitecture
    from ..config import PipelineConfig


@dataclass(frozen=True)
class PartitionSpec:
    """Where a checkpoint's decoder may be cut into pipeline stages.

    An architecture config that runs as pipeline stages returns one by
    implementing :class:`ArchConfigWithPartitionSpec`.
    """

    num_layers: int
    """The number of decoder layers."""

    forbidden_cuts: frozenset[int] = frozenset()
    """Layer ids no stage may start at, such as a layer that reads a cache an
    earlier layer writes."""


@runtime_checkable
class ArchConfigWithPartitionSpec(Protocol):
    """An architecture config whose decoder may be cut into pipeline stages."""

    @classmethod
    def construct_partition_spec(
        cls, huggingface_config: AutoConfig
    ) -> PartitionSpec:
        """Returns where the checkpoint's decoder may be cut.

        Raises:
            ValueError: If the checkpoint cannot be split into pipeline
                stages.
        """
        ...


def partition_spec(
    arch: SupportedArchitecture, huggingface_config: AutoConfig
) -> PartitionSpec:
    """Returns where the architecture allows the checkpoint to be cut.

    Args:
        arch: The model's architecture.
        huggingface_config: The checkpoint's config.

    Raises:
        TypeError: If the architecture's config does not implement
            :class:`ArchConfigWithPartitionSpec`.
        ValueError: If the checkpoint cannot be split into pipeline stages.
    """
    config = arch.config
    if not issubclass(config, ArchConfigWithPartitionSpec):
        raise TypeError(
            f"{arch.name} does not run as pipeline stages, since"
            f" {config.__name__} does not implement"
            " ArchConfigWithPartitionSpec."
        )
    return config.construct_partition_spec(huggingface_config)


def split_layers(
    spec: PartitionSpec, num_stages: int, sizes: list[int] | None = None
) -> list[int]:
    """Returns each pipeline stage's decoder layer count.

    Args:
        spec: Where the decoder may be cut.
        num_stages: The number of pipeline stages.
        sizes: Layer counts to validate. When ``None``, the split closest
            to an even one, with the remainder on the earliest stages, whose
            cuts the spec allows.

    Raises:
        ValueError: If ``num_stages`` is not positive, ``sizes`` is
            invalid, or no allowed cuts leave every stage a layer.
    """
    if num_stages < 1:
        raise ValueError(
            f"A pipeline needs at least one stage, got {num_stages}."
        )
    n = spec.num_layers
    if sizes is None:
        base, extra = divmod(n, num_stages)
        even = [
            stage * base + min(stage, extra) for stage in range(1, num_stages)
        ]
        starts = [0, *_closest_cuts(spec, even)]
        ends = [*starts[1:], n]
        sizes = [end - start for start, end in zip(starts, ends, strict=True)]
    if len(sizes) != num_stages:
        raise ValueError(
            f"The pipeline partition {sizes} has {len(sizes)} stages, but the"
            f" pipeline has {num_stages}."
        )
    if any(size < 1 for size in sizes):
        raise ValueError(f"Every pipeline stage needs a layer, got {sizes}.")
    if sum(sizes) != n:
        raise ValueError(
            f"The pipeline partition {sizes} covers {sum(sizes)} decoder"
            f" layers, but the model has {n}."
        )
    start = 0
    for size in sizes[:-1]:
        start += size
        if start in spec.forbidden_cuts:
            raise ValueError(
                f"The pipeline partition {sizes} starts a stage at layer"
                f" {start}, where the architecture forbids a cut."
            )
    return list(sizes)


def _closest_cuts(spec: PartitionSpec, even: list[int]) -> list[int]:
    """Returns the allowed stage starts closest in total to ``even``.

    The cuts are chosen together, since moving one off a forbidden layer can
    leave the next with no room. Of the closest, the earliest wins.

    Raises:
        ValueError: If no allowed cuts leave every stage a layer.
    """
    n = spec.num_layers
    allowed = [c for c in range(1, n) if c not in spec.forbidden_cuts]
    # cost[k][i] is the least total distance of cut k onward when cut k sits
    # at allowed[i], and each later cut sits further on.
    cost = [[math.inf] * len(allowed) for _ in even]
    for k in reversed(range(len(even))):
        for i, cut in enumerate(allowed):
            rest = (
                min(cost[k + 1][i + 1 :], default=math.inf)
                if k + 1 < len(even)
                else 0
            )
            cost[k][i] = abs(cut - even[k]) + rest
    cuts: list[int] = []
    first = 0
    for k in range(len(even)):
        best, i = min(
            ((cost[k][i], i) for i in range(first, len(allowed))),
            default=(math.inf, -1),
        )
        if best == math.inf:
            raise ValueError(
                f"No allowed cut leaves {len(even) + 1} non-empty stages over"
                f" {n} layers."
            )
        cuts.append(allowed[i])
        first = i + 1
    return cuts


def pipeline_stages(sizes: list[int]) -> list[PipelineStage]:
    """Returns the stages of a partition, in order.

    Args:
        sizes: Each stage's layer count, as :func:`split_layers` returns.
    """
    stages = []
    start = 0
    for index, size in enumerate(sizes):
        stages.append(
            PipelineStage(index, len(sizes), range(start, start + size))
        )
        start += size
    return stages


def stage_layers(pipeline_config: PipelineConfig, num_layers: int) -> range:
    """Returns the decoder layers the config builds.

    Every layer, unless the config carries a pipeline stage.

    Args:
        pipeline_config: The config a model builds from.
        num_layers: The decoder's layer count, checked against the stage.

    Raises:
        ValueError: If the stage's layers do not fit the decoder.
    """
    stage = pipeline_config.model.pipeline_stage
    if stage is None:
        return range(num_layers)
    layers = stage.layers
    if layers.stop > num_layers or stage.is_last != (layers.stop == num_layers):
        raise ValueError(
            f"Pipeline stage {stage.index} of {stage.num_stages} holds layers"
            f" {layers.start} to {layers.stop - 1}, which does not fit a"
            f" decoder of {num_layers} layers."
        )
    return layers


def stage_pipeline_config(
    pipeline_config: PipelineConfig, stage: PipelineStage
) -> PipelineConfig:
    """Returns the config that builds one pipeline stage of the main model.

    Args:
        pipeline_config: The config that builds the whole model.
        stage: The stage to build.
    """
    stage_model = pipeline_config.model.model_copy()
    stage_model._pipeline_stage = stage
    models = pipeline_config.models.with_override("main", config=stage_model)
    return pipeline_config.model_copy(update={"models": models})
