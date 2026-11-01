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
"""Registers Idefics3 with the ModuleV3 registry."""

from max.graph.weights import WeightsFormat
from max.pipelines.architectures.idefics3.tokenizer import Idefics3Tokenizer
from max.pipelines.context import TextAndVisionContext
from max.pipelines.kv_cache.memory_planner import PagedMemoryPlanner
from max.pipelines.lib import GraphRole, SupportedArchitecture
from max.pipelines.modeling.types import InputModality, PipelineTask

from .batch_processor import Idefics3ModuleV3BatchProcessor
from .model_config import Idefics3Config
from .modules import Idefics3Decoder, Idefics3Vision
from .weight_adapters import convert_registered_state_dict

idefics3_modulev3_arch = SupportedArchitecture(
    name="Idefics3ForConditionalGeneration_ModuleV3",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["HuggingFaceM4/Idefics3-8B-Llama3"],
    default_encoding=Idefics3Config.DEFAULT_ENCODING,
    supported_encodings=Idefics3Config.SUPPORTED_ENCODINGS,
    modules={
        GraphRole.DECODER: Idefics3Decoder,
        GraphRole.VISION: Idefics3Vision,
    },
    tokenizer=Idefics3Tokenizer,
    context_type=TextAndVisionContext,
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=False,
    input_modalities={InputModality.TEXT, InputModality.IMAGE},
    required_arguments={
        "enable_chunked_prefill": False,
        "enable_prefix_caching": False,
    },
    config=Idefics3Config,
    weight_adapters={WeightsFormat.safetensors: convert_registered_state_dict},
    batching=Idefics3ModuleV3BatchProcessor,
    memory_planner=PagedMemoryPlanner,
    supports_overlap_scheduler=False,
    supports_device_graph_capture=False,
)
