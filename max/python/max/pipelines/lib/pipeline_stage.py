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

"""Defines :class:`PipelineStage`, one stage of a decoder cut into stages.

It imports nothing else from :mod:`max.pipelines`, so a model config can carry
one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from max.graph import TensorType
from max.tree import Tree


@dataclass(frozen=True)
class PipelineStage:
    """One stage of a decoder cut into pipeline stages."""

    index: int
    """The stage's position in the pipeline, from zero."""

    num_stages: int
    """The number of stages in the pipeline."""

    layers: range
    """The decoder layers the stage builds."""

    boundary_types: Tree[TensorType] | None = field(default=None, hash=False)
    """The types of what the previous stage's decoder returns, in the tree it
    returns them, or ``None`` on the first stage."""

    trace_only: bool = False
    """Whether the build ends with
    :class:`~max.pipelines.lib.pipeline_parallel.StageTraced` once the stage knows
    what it hands the next, before anything compiles. An exporter traces the
    stages before the one it builds, to learn that stage's boundary types."""

    @property
    def is_first(self) -> bool:
        """Whether the stage takes the model's inputs rather than a cut's."""
        return self.index == 0

    @property
    def is_last(self) -> bool:
        """Whether the stage returns the model's outputs rather than a cut's."""
        return self.index == self.num_stages - 1
