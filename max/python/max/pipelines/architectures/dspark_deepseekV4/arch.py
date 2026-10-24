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

from ..deepseekV4 import weight_adapters
from ..deepseekV4.model_config import DeepseekV4Config
from .model import (
    DSparkDeepseekV4DraftArchConfig,
    dspark_draft_width,
)

__all__ = ["dspark_deepseekV4_draft_arch"]

# The stages ship in the trunk's checkpoint, which has no config of their
# own, so spec decoding names this architecture from the target's. It has
# no pipeline model of its own.
dspark_deepseekV4_draft_arch = SupportedArchitecture(
    name="DSparkDeepseekV4DraftModel",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["deepseek-ai/DeepSeek-V4-Flash-0731"],
    default_encoding=DeepseekV4Config.DEFAULT_ENCODING,
    supported_encodings=DeepseekV4Config.SUPPORTED_ENCODINGS,
    # Split over the trunk's devices, as the trunk is.
    multi_gpu_supported=True,
    pipeline_model=None,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    default_weights_format=WeightsFormat.safetensors,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_dspark_safetensor_state_dict,
    },
    config=DSparkDeepseekV4DraftArchConfig,
    checkpoint_draft_width=dspark_draft_width,
    # The base captures its decode step; the spec-decode graph hasn't been
    # validated under capture yet.
    supports_device_graph_capture=False,
)
