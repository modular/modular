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
from max.interfaces import InputModality, PipelineTask
from max.pipelines.lib import SupportedArchitecture

from .context import Gemma4Context
from .model import Gemma3_MultiModalModel
from .model_config import Gemma4ForConditionalGenerationConfig
from .tokenizer import Gemma4Tokenizer

example_repo_ids = [
    # it = Instruction tuned (recommended).
    # pt = Pre-trained.
    "google/gemma-4-31B-it",
    # "google/gemma-4-26B-A4B-it"
]

gemma4_arch = SupportedArchitecture(
    name="Gemma4ForConditionalGeneration",
    example_repo_ids=example_repo_ids,
    default_encoding="bfloat16",
    supported_encodings={
        "bfloat16",
    },
    pipeline_model=Gemma3_MultiModalModel,
    task=PipelineTask.TEXT_GENERATION,
    tokenizer=Gemma4Tokenizer,
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=False,
    input_modalities={
        InputModality.TEXT,
        InputModality.IMAGE,
        InputModality.VIDEO,
    },
    rope_type="normal",
    required_arguments={"max_num_steps": 1},
    context_type=Gemma4Context,
    config=Gemma4ForConditionalGenerationConfig,
    tool_parser="gemma4",
)
