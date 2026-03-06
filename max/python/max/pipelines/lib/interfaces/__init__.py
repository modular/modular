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

from .arch_config import (
    ArchConfig,
    ArchConfigWithAttentionKVCache,
    ArchConfigWithKVCache,
)
from .component_model import ComponentModel
from .diffusion_pipeline import DiffusionPipeline, PixelModelInputs
from .generate import GenerateMixin
from .pipeline_model import (
    AlwaysSignalBuffersMixin,
    ModelInputs,
    ModelOutputs,
    PipelineModel,
    PipelineModelWithKVCache,
)

__all__ = [
    "AlwaysSignalBuffersMixin",
    "ArchConfig",
    "ArchConfigWithAttentionKVCache",
    "ArchConfigWithKVCache",
    "ComponentModel",
    "DiffusionPipeline",
    "GenerateMixin",
    "ModelInputs",
    "ModelOutputs",
    "PipelineModel",
    "PipelineModelWithKVCache",
    "PixelModelInputs",
    "get_paged_manager",
]
