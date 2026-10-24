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
from max.pipelines.lib import SupportedArchitecture, TextTokenizer
from max.pipelines.modeling.types import PipelineTask

from ..deepseekV3.model_config import DeepseekV3Config
from .model import MTPDeepseekV3DraftArchConfig
from .weight_adapters import convert_mtp_safetensor_state_dict

__all__ = ["mtp_deepseekV3_draft_arch"]

# The NextN head ships in the target's checkpoint, which has no config of its
# own, so spec decoding names this architecture from the target's. It has
# no pipeline model of its own.
mtp_deepseekV3_draft_arch = SupportedArchitecture(
    name="MTPDeepseekV3DraftModel",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["deepseek-ai/DeepSeek-V3"],
    default_encoding=DeepseekV3Config.DEFAULT_ENCODING,
    supported_encodings=DeepseekV3Config.SUPPORTED_ENCODINGS,
    # Split over the target's devices, as the target is.
    multi_gpu_supported=True,
    pipeline_model=None,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    default_weights_format=WeightsFormat.safetensors,
    weight_adapters={
        WeightsFormat.safetensors: convert_mtp_safetensor_state_dict,
    },
    config=MTPDeepseekV3DraftArchConfig,
)
