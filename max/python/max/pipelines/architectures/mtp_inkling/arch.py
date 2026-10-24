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
from max.pipelines.lib import SupportedArchitecture
from max.pipelines.modeling.types import InputModality, PipelineTask

from ..inkling.model_config import InklingConfig
from ..inkling.tokenizer import InklingTokenizer
from .model import (
    MTPInklingDraftArchConfig,
    inkling_mtp_draft_width,
)
from .weight_adapters import convert_inkling_mtp_state_dict

__all__ = ["mtp_inkling_draft_arch"]

# The MTP depths ship in the target's checkpoint, which has no config of its
# own, so spec decoding names this architecture from the target's. It has
# no pipeline model of its own.
mtp_inkling_draft_arch = SupportedArchitecture(
    name="MTPInklingDraftModel",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["thinkingmachines/Inkling"],
    default_encoding=InklingConfig.DEFAULT_ENCODING,
    supported_encodings=InklingConfig.SUPPORTED_ENCODINGS,
    # Split over the target's devices, as the target is.
    multi_gpu_supported=True,
    supports_device_graph_capture=True,
    pipeline_model=None,
    tokenizer=InklingTokenizer,
    context_type=TextAndVisionContext,
    default_weights_format=WeightsFormat.safetensors,
    weight_adapters={
        WeightsFormat.safetensors: convert_inkling_mtp_state_dict,
    },
    config=MTPInklingDraftArchConfig,
    checkpoint_draft_width=inkling_mtp_draft_width,
    input_modalities={InputModality.TEXT, InputModality.IMAGE},
)
