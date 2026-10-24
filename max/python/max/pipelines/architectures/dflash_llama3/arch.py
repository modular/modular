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
from max.pipelines.speculative._dflash import dflash_draft_width

from ..llama3 import weight_adapters
from .model import DFlashLlama3ArchConfig

# The DFlash draft checkpoint's architecture. It has no pipeline model: its
# arch config builds the draft for a unified spec-decode graph, and the width
# it was trained for comes from its checkpoint.
dflash_llama_arch = SupportedArchitecture(
    name="DFlashDraftModel",
    example_repo_ids=[
        "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat",
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
    multi_gpu_supported=False,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
    task=PipelineTask.TEXT_GENERATION,
    config=DFlashLlama3ArchConfig,
    supports_overlap_scheduler=False,
    supports_device_graph_capture=False,
    checkpoint_draft_width=dflash_draft_width,
)
