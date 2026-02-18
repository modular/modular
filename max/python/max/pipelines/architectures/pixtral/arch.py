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
from max.pipelines.core.context_validators import (
    validate_only_one_image,
    validate_requires_vision_context,
)
from max.pipelines.lib import (
    SupportedArchitecture,
    SupportedEncoding,
    TextAndVisionTokenizer,
)

from . import weight_adapters
from .model import PixtralModel
from .model_config import PixtralConfig

pixtral_arch = SupportedArchitecture(
    name="LlavaForConditionalGeneration_Legacy",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["mistral-community/pixtral-12b"],
    default_encoding=SupportedEncoding.bfloat16,
    supported_encodings={
        SupportedEncoding.bfloat16: ["paged"],
    },
    pipeline_model=PixtralModel,
    tokenizer=TextAndVisionTokenizer,
    context_type=TextAndVisionContext,
    default_weights_format=WeightsFormat.safetensors,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
    required_arguments={
        "enable_prefix_caching": False,
        "enable_chunked_prefill": False,
    },
    context_validators=[
        validate_requires_vision_context,
        validate_only_one_image,
    ],
    config=PixtralConfig,
)
