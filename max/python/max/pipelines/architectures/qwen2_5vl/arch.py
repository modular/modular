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
from max.pipelines.core import TextAndVisionContext, TextContext
from max.pipelines.core.exceptions import InputError
from max.pipelines.lib import SupportedArchitecture

from .context import Qwen2_5VLTextAndVisionContext
from .model import Qwen2_5VLModel
from .model_config import Qwen2_5VLConfig
from .tokenizer import Qwen2_5VLTokenizer
from .weight_adapters import convert_qwen2_5vl_model_state_dict


def validate_qwen2_5vl_required_args(
    context: TextContext | TextAndVisionContext,
) -> None:
    """Validates that all required Qwen2.5VL arguments are present.

    Checks that the context is a Qwen2_5VLTextAndVisionContext with the
    required direct fields (rope_delta, decoder_position_ids) and, when
    vision encoding is needed, that vision_data is populated.

    Args:
        context: The context to validate.

    Raises:
        InputError: If the context is not the expected type or is missing
            required fields.
    """
    if not isinstance(context, Qwen2_5VLTextAndVisionContext):
        raise InputError(
            f"context must be Qwen2_5VLTextAndVisionContext, got {type(context).__name__}"
        )

    # Required only when vision encoding is needed
    if context.needs_vision_encoding and context.vision_data is None:
        raise InputError(
            "vision_data is required for Qwen2.5VL when vision encoding is needed"
        )


qwen2_5_vl_arch = SupportedArchitecture(
    name="Qwen2_5_VLForConditionalGeneration",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=[
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "Qwen/Qwen2.5-VL-7B-Instruct",
    ],
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=True,
    default_encoding="bfloat16",
    supported_encodings={
        "float32",
        "bfloat16",
        "float8_e4m3fn",
    },
    weight_adapters={
        WeightsFormat.safetensors: convert_qwen2_5vl_model_state_dict,
    },
    pipeline_model=Qwen2_5VLModel,
    tokenizer=Qwen2_5VLTokenizer,
    context_type=Qwen2_5VLTextAndVisionContext,
    required_arguments={
        "enable_chunked_prefill": False,
    },
    context_validators=[
        validate_qwen2_5vl_required_args,
    ],
    config=Qwen2_5VLConfig,
)
