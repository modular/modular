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

"""Pipeline parallelism: a decoder cut into stages of contiguous layers."""

from .contract import (
    PipelinedModel,
    StageTraced,
    raise_stage_traced,
    stage_graph,
    stage_input_types,
    stage_outputs,
)
from .partition import (
    ArchConfigWithPartitionSpec,
    PartitionSpec,
    PipelineStage,
    partition_spec,
    pipeline_stages,
    split_layers,
    stage_layers,
    stage_pipeline_config,
)

__all__ = [
    "ArchConfigWithPartitionSpec",
    "PartitionSpec",
    "PipelineStage",
    "PipelinedModel",
    "StageTraced",
    "partition_spec",
    "pipeline_stages",
    "raise_stage_traced",
    "split_layers",
    "stage_graph",
    "stage_input_types",
    "stage_layers",
    "stage_outputs",
    "stage_pipeline_config",
]
