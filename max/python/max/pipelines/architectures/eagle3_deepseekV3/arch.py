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

from dataclasses import replace

from max.graph.weights import WeightsFormat
from max.pipelines.context import TextContext
from max.pipelines.lib import SupportedArchitecture, TextTokenizer
from max.pipelines.modeling.types import InputModality, PipelineTask

from ..deepseekV3.model_config import DeepseekV3Config
from ..kimik2_5.weight_adapters import (
    convert_eagle3_draft_state_dict,
    convert_llama_eagle3_draft_state_dict,
)
from .model import (
    Eagle3DeepseekV3DraftArchConfig,
    Eagle3MHADeepseekV3DraftArchConfig,
)

# The MLA head's checkpoint names Kimi K2.5's fused Eagle3 architecture, so
# spec decoding renames it to this one beside a DeepseekV3 target. It has
# no pipeline model of its own.
eagle3_deepseekV3_draft_arch = SupportedArchitecture(
    name="Eagle3DeepseekV3DraftModel",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["nvidia/Kimi-K2.5-Thinking-Eagle3"],
    default_encoding=DeepseekV3Config.DEFAULT_ENCODING,
    supported_encodings=DeepseekV3Config.SUPPORTED_ENCODINGS,
    # Split over the target's devices, as the target is.
    multi_gpu_supported=True,
    pipeline_model=None,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    default_weights_format=WeightsFormat.safetensors,
    weight_adapters={
        WeightsFormat.safetensors: convert_eagle3_draft_state_dict,
    },
    config=Eagle3DeepseekV3DraftArchConfig,
    input_modalities={InputModality.TEXT, InputModality.IMAGE},
)

# The Llama-style MHA head's checkpoint names Llama 3's Eagle3 draft, so
# spec decoding renames it to this one beside a DeepseekV3 target.
eagle3_mha_deepseekV3_draft_arch = replace(
    eagle3_deepseekV3_draft_arch,
    name="Eagle3MHADeepseekV3DraftModel",
    example_repo_ids=["modularai/kimi-k2.5-eagle3"],
    pipeline_model=None,
    weight_adapters={
        WeightsFormat.safetensors: convert_llama_eagle3_draft_state_dict,
    },
    config=Eagle3MHADeepseekV3DraftArchConfig,
)
