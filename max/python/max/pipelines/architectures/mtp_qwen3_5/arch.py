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

from ..qwen3_5.model_config import Qwen3_5Config
from .model import MTPQwen3_5DraftArchConfig

__all__ = ["mtp_qwen3_5_draft_arch"]

# The MTP head ships in the target's checkpoint, which has no config of its
# own, so spec decoding names this architecture from the target's. It has
# no pipeline model of its own.
mtp_qwen3_5_draft_arch = SupportedArchitecture(
    name="MTPQwen3_5DraftModel",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["RadixArk/Qwen3.8-27B-NVFP4"],
    default_encoding=Qwen3_5Config.DEFAULT_ENCODING,
    supported_encodings=Qwen3_5Config.SUPPORTED_ENCODINGS,
    # Split over the target's devices, as the target is.
    multi_gpu_supported=True,
    pipeline_model=None,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    default_weights_format=WeightsFormat.safetensors,
    config=MTPQwen3_5DraftArchConfig,
    supports_device_graph_capture=False,
)
