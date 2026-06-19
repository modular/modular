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
"""Interfaces for MAX pipelines."""

from max.pipelines.diffusion.interface import (
    DiffusionPipeline,
    DiffusionPipelineOutput,
)

from .arch_config import (
    ArchConfig,
    ArchConfigWithAttentionKVCache,
    ArchConfigWithBoundedMaxSeqLen,
    ArchConfigWithKVCache,
    ArchConfigWithPermissiveMaxSeqLen,
    ArchConfigWithStoredKVParams,
    ArchVLConfigWithTextSubconfig,
)
from .batch_processor import (
    BatchProcessor,
    BatchProcessorRuntime,
    RaggedBatchProcessor,
    process_ragged_kv_outputs,
    ragged_kv_symbolic_inputs,
)
from .generate import GenerateMixin
from .pipeline_model import (
    AlwaysSignalBuffersMixin,
    ModelInputs,
    ModelOutputs,
    PipelineModel,
    PipelineModelWithKVCache,
    UnifiedEagleOutputs,
)

__all__ = [
    "AlwaysSignalBuffersMixin",
    "ArchConfig",
    "ArchConfigWithAttentionKVCache",
    "ArchConfigWithBoundedMaxSeqLen",
    "ArchConfigWithKVCache",
    "ArchConfigWithPermissiveMaxSeqLen",
    "ArchConfigWithStoredKVParams",
    "ArchVLConfigWithTextSubconfig",
    "BatchProcessor",
    "BatchProcessorRuntime",
    "DiffusionPipeline",
    "DiffusionPipelineOutput",
    "GenerateMixin",
    "ModelInputs",
    "ModelOutputs",
    "PipelineModel",
    "PipelineModelWithKVCache",
    "RaggedBatchProcessor",
    "UnifiedEagleOutputs",
    "process_ragged_kv_outputs",
    "ragged_kv_symbolic_inputs",
]
