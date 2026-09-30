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
from max.pipelines.context import TextContext
from max.pipelines.lib import SupportedArchitecture
from max.pipelines.modeling.types import PipelineTask

# Nemotron-3's chat template is the Qwen3.5 one: an implicit ``<think>`` open
# and Qwen3-Coder XML tool calls. Importing the parser modules registers the
# parsers named below.
from ..qwen3_5.reasoning import (
    Qwen3_5ReasoningParser,  # noqa: F401
)
from ..qwen3_5.tool_parser import (
    Qwen3_5ToolParser,  # noqa: F401
)
from .memory_planner import NemotronHMemoryPlanner
from .model import NemotronHModel
from .model_config import NemotronHConfig
from .tokenizer import NemotronHTokenizer
from .weight_adapters import convert_nemotron_h_state_dict

nemotron_h_modulev3_arch = SupportedArchitecture(
    name="NemotronHForCausalLM_ModuleV3",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=[
        "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
        "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
    ],
    default_weights_format=WeightsFormat.safetensors,
    default_encoding=NemotronHConfig.DEFAULT_ENCODING,
    supported_encodings=NemotronHConfig.SUPPORTED_ENCODINGS,
    pipeline_model=NemotronHModel,
    tokenizer=NemotronHTokenizer,
    context_type=TextContext,
    weight_adapters={
        WeightsFormat.safetensors: convert_nemotron_h_state_dict,
    },
    checkpoints_recurrent_state=True,
    config=NemotronHConfig,
    batching=NemotronHModel.batch_processor_cls,
    memory_planner=NemotronHMemoryPlanner,
    reasoning_parser="qwen3_5",
    tool_parser="qwen3_5",
)
