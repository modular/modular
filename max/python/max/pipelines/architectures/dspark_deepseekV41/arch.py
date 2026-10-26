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
from max.pipelines.context import TextAndVisionContext
from max.pipelines.lib import SupportedArchitecture, TextTokenizer
from max.pipelines.modeling.types import InputModality, PipelineTask

from ..deepseekV41 import weight_adapters
from ..deepseekV41.model_config import DeepseekV41Config, dspark_draft_width
from .model import DSparkDeepseekV41DraftArchConfig

__all__ = ["dspark_deepseekV41_draft_arch"]

# The stages ship in the trunk's checkpoint, which has no config of their
# own, so spec decoding names this architecture from the target's. It has
# no pipeline model of its own.
dspark_deepseekV41_draft_arch = SupportedArchitecture(
    name="DSparkDeepseekV41DraftModel",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["nvidia/DeepSeek-V4.1-Flash-NVFP4"],
    default_encoding=DeepseekV41Config.DEFAULT_ENCODING,
    supported_encodings=DeepseekV41Config.SUPPORTED_ENCODINGS,
    # Split over the trunk's devices, as the trunk is.
    multi_gpu_supported=True,
    pipeline_model=None,
    tokenizer=TextTokenizer,
    # The stages draft from the trunk's states, image spans included.
    context_type=TextAndVisionContext,
    input_modalities={InputModality.TEXT, InputModality.IMAGE},
    default_weights_format=WeightsFormat.safetensors,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_dspark_safetensor_state_dict,
    },
    config=DSparkDeepseekV41DraftArchConfig,
    checkpoint_draft_width=dspark_draft_width,
    # One captured graph per (batch size, verification width, cache-length
    # bucket); every input, including the pinned Engram tables, is bound as
    # a buffer so a replay sees fresh values.
    supports_device_graph_capture=True,
)
