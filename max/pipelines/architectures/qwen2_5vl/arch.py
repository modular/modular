# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
from max.nn.kv_cache import KVCacheStrategy
from max.pipelines.core import TextAndVisionContext, TextContext
from max.pipelines.core.exceptions import InputError
from max.pipelines.lib import SupportedArchitecture, SupportedEncoding

from .model import Qwen2_5VLModel
from .tokenizer import Qwen2_5VLTokenizer
from .weight_adapters import convert_qwen2_5vl_model_state_dict


def validate_qwen2_5vl_required_args(
    context: TextContext | TextAndVisionContext,
) -> None:
    """Validates that all required Qwen2.5VL arguments are present.

    Args:
        context: The context to validate.

    Raises:
        InputError: If required arguments are missing from extra_model_args.
    """
    if not isinstance(context, TextAndVisionContext):
        raise ValueError(f"context must be TextAndVisionContext, got {context}")

    # Always required for Qwen2.5VL
    required_always = ["rope_delta", "decoder_position_ids"]

    for arg in required_always:
        if arg not in context.extra_model_args:
            raise InputError(
                f"{arg} is required in extra_model_args for Qwen2.5VL"
            )

    # Required only when vision encoding is needed
    if context.needs_vision_encoding:
        required_for_vision = [
            "vision_position_ids",
            "window_index",
            "cu_seqlens",
            "cu_window_seqlens_unique",
            "max_seqlen",
            "window_max_seqlen",
            "max_grid_size",
        ]

        for arg in required_for_vision:
            if arg not in context.extra_model_args:
                raise InputError(
                    f"{arg} is required in extra_model_args for Qwen2.5VL when vision encoding is needed"
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
    default_encoding=SupportedEncoding.bfloat16,
    supported_encodings={
        SupportedEncoding.float32: [KVCacheStrategy.PAGED],
        SupportedEncoding.bfloat16: [KVCacheStrategy.PAGED],
    },
    weight_adapters={
        WeightsFormat.safetensors: convert_qwen2_5vl_model_state_dict,
    },
    pipeline_model=Qwen2_5VLModel,
    tokenizer=Qwen2_5VLTokenizer,
    required_arguments={
        "enable_prefix_caching": False,
        "enable_chunked_prefill": False,
    },
    context_validators=[
        validate_qwen2_5vl_required_args,
    ],
)
