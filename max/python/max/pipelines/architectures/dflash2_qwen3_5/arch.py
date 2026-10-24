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
from max.pipelines.lib import (
    SupportedArchitecture,
    TextTokenizer,
)
from max.pipelines.modeling.types import PipelineTask

from ..llama3 import weight_adapters
from .model import DFlash2Qwen3_5ArchConfig
from .model_config import dflash2_draft_width

# The DFlash2 draft checkpoint's architecture. It has no pipeline model: its
# arch config builds the draft for a unified spec-decode graph, and the width
# it was trained for comes from its checkpoint.
dflash2_qwen3_5_arch = SupportedArchitecture(
    name="DFlash2DraftModel",
    example_repo_ids=[
        "z-lab/Qwen3.8-27B-DFlash2",
    ],
    default_encoding="bfloat16",
    supported_encodings={
        "bfloat16",
        "float32",
    },
    pipeline_model=None,
    context_type=TextContext,
    tokenizer=TextTokenizer,
    default_weights_format=WeightsFormat.safetensors,
    # The draft body is single-device; DFlash2Qwen3_5 enforces that.
    multi_gpu_supported=False,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
    task=PipelineTask.TEXT_GENERATION,
    config=DFlash2Qwen3_5ArchConfig,
    supports_overlap_scheduler=False,
    supports_device_graph_capture=False,
    checkpoint_draft_width=dflash2_draft_width,
)
