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
from max.interfaces import PipelineTask
from max.pipelines.core import TextAndVisionContext
from max.pipelines.lib import (
    SupportedArchitecture,
    TextAndVisionTokenizer,
)

from .model import Gemma3_MultiModalModel
from .model_config import Gemma3ForConditionalGenerationConfig

example_repo_ids = [
    # it = Instruction tuned (recommended).
    # pt = Pre-trained.
    "google/gemma-3-12b-it",
    "google/gemma-3-12b-pt",
    "google/gemma-3-4b-it",
    "google/gemma-3-4b-pt",
    "google/gemma-3-12b-it",
    "google/gemma-3-12b-pt",
    "google/gemma-3-27b-it",
    "google/gemma-3-27b-pt",
]

gemma3_multimodal_arch = SupportedArchitecture(
    name="Gemma3ForConditionalGeneration",
    example_repo_ids=example_repo_ids,
    default_encoding="bfloat16",
    supported_encodings={
        "bfloat16": ["paged"],
        "float8_e4m3fn": ["paged"],
    },
    pipeline_model=Gemma3_MultiModalModel,
    task=PipelineTask.TEXT_GENERATION,
    tokenizer=TextAndVisionTokenizer,
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=True,
    rope_type="normal",
    required_arguments={
        "enable_prefix_caching": False,
        "enable_chunked_prefill": False,
    },
    context_type=TextAndVisionContext,
    config=Gemma3ForConditionalGenerationConfig,
)
