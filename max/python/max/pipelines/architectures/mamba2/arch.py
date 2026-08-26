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

from __future__ import annotations

from max.graph.weights import WeightsFormat
from max.pipelines.context import TextContext
from max.pipelines.lib import SupportedArchitecture
from max.pipelines.modeling.types import PipelineTask

from . import weight_adapters
from .batch_processor import Mamba2BatchProcessor
from .memory_planner import Mamba2MemoryPlanner
from .model import Mamba2Model
from .model_config import Mamba2ArchConfig
from .tokenizer import Mamba2Tokenizer

mamba2_arch = SupportedArchitecture(
    name="Mamba2ForCausalLM",
    example_repo_ids=[
        "state-spaces/mamba2-130m",
        "state-spaces/mamba2-370m",
        "state-spaces/mamba2-780m",
        "state-spaces/mamba2-1.3b",
        "state-spaces/mamba2-2.7b",
    ],
    default_encoding="float32",
    supported_encodings={
        "float32",
        "bfloat16",
    },
    pipeline_model=Mamba2Model,
    batching=Mamba2BatchProcessor,
    tokenizer=Mamba2Tokenizer,
    context_type=TextContext,
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=False,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_mamba2_state_dict,
    },
    task=PipelineTask.TEXT_GENERATION,
    config=Mamba2ArchConfig,
    memory_planner=Mamba2MemoryPlanner,
    supports_overlap_scheduler=False,
    supports_device_graph_capture=False,
)
