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
"""Registers the HY-V4 MTP draft."""

from max.graph.weights import WeightsFormat
from max.pipelines.context import TextContext
from max.pipelines.lib import SupportedArchitecture
from max.pipelines.modeling.types import PipelineTask

from ..hy_v4.model_config import HYV4Config
from ..hy_v4.tokenizer import HYV4Tokenizer
from .model import MTPHYV4DraftArchConfig
from .weight_adapters import convert_hy_v4_mtp_state_dict

__all__ = ["mtp_hy_v4_draft_arch"]

# The MTP layer ships in the target's checkpoint, which has no config of its
# own, so spec decoding names this architecture from the target's. It has no
# pipeline model of its own.
mtp_hy_v4_draft_arch = SupportedArchitecture(
    name="MTPHYV4DraftModel",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["tencent/Hy4-preview-FP8"],
    default_encoding=HYV4Config.DEFAULT_ENCODING,
    supported_encodings=HYV4Config.SUPPORTED_ENCODINGS,
    # Split over the target's devices, as the target is.
    multi_gpu_supported=True,
    pipeline_model=None,
    tokenizer=HYV4Tokenizer,
    context_type=TextContext,
    default_weights_format=WeightsFormat.safetensors,
    weight_adapters={
        WeightsFormat.safetensors: convert_hy_v4_mtp_state_dict,
    },
    config=MTPHYV4DraftArchConfig,
)
