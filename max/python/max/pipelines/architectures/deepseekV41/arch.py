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

from max.graph.weights import WeightsFormat
from max.pipelines.architectures.deepseekV4.reasoning import (  # noqa: F401
    DeepseekV4ReasoningParser,
)
from max.pipelines.context import TextAndVisionContext
from max.pipelines.lib import SupportedArchitecture
from max.pipelines.modeling.types import InputModality, PipelineTask

from . import weight_adapters
from .memory_planner import DeepseekV41MemoryPlanner
from .model import DeepseekV41Model
from .model_config import DeepseekV41Config, dspark_draft_width
from .tokenizer import DeepseekV41Tokenizer
from .tool_parser import DeepseekV41ToolParser  # noqa: F401

deepseekV41_arch = SupportedArchitecture(
    name="DeepseekV41ForCausalLM",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=[
        "nvidia/DeepSeek-V4.1-Flash-NVFP4",
    ],
    default_encoding=DeepseekV41Config.DEFAULT_ENCODING,
    supported_encodings=DeepseekV41Config.SUPPORTED_ENCODINGS,
    multi_gpu_supported=True,
    pipeline_model=DeepseekV41Model,
    tokenizer=DeepseekV41Tokenizer,
    context_type=TextAndVisionContext,
    input_modalities={InputModality.TEXT, InputModality.IMAGE},
    default_weights_format=WeightsFormat.safetensors,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
    config=DeepseekV41Config,
    memory_planner=DeepseekV41MemoryPlanner,
    # Speculative decoding drafts this many tokens per step, so the KV cache
    # reserves room for them.
    checkpoint_draft_width=dspark_draft_width,
    reasoning_parser="deepseekv4",
    tool_parser="deepseekv41",
)
