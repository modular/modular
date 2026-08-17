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

import dataclasses

from max.graph.weights import WeightsFormat
from max.pipelines.architectures.gemma4.model_config import (
    Gemma4ForConditionalGenerationConfig,
)
from max.pipelines.context import TextContext
from max.pipelines.kv_cache.memory_planner import PagedMemoryPlanner
from max.pipelines.lib import SupportedArchitecture, TextTokenizer
from max.pipelines.modeling.types import PipelineTask

from . import weight_adapters
from .batch_processor import Gemma4ModuleV3BatchProcessor
from .model import Gemma4Model

gemma4_modulev3_arch = SupportedArchitecture(
    name="Gemma4ForConditionalGeneration_ModuleV3",
    example_repo_ids=["google/gemma-4-31B-it"],
    default_encoding="bfloat16",
    supported_encodings={"bfloat16"},
    pipeline_model=Gemma4Model,
    task=PipelineTask.TEXT_GENERATION,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=False,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
    config=Gemma4ForConditionalGenerationConfig,
    batching=Gemma4ModuleV3BatchProcessor,
    memory_planner=PagedMemoryPlanner,
    supports_overlap_scheduler=False,
    supports_device_graph_capture=False,
)

# Text-only model_type "gemma4_unified" line (e.g. google/gemma-4-12B-it):
# same model, different HF architecture string -- mirrors the graph side's
# dataclasses.replace at gemma4/arch.py:70-77.
gemma4_unified_modulev3_arch = dataclasses.replace(
    gemma4_modulev3_arch,
    name="Gemma4UnifiedForConditionalGeneration_ModuleV3",
    example_repo_ids=["google/gemma-4-12B-it"],
)
