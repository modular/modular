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
from max.pipelines.modeling.types import InputModality, PipelineTask
from max.pipelines.speculative._dflash import dflash_draft_width

from .model import DFlashKimiK25DraftArchConfig

__all__ = ["dflash_kimi_k25_draft_arch"]

# The checkpoint names Llama 3's DFlash draft, so spec decoding renames it
# to this one beside a Kimi K2.5 target. It has no
# pipeline model of its own.
dflash_kimi_k25_draft_arch = SupportedArchitecture(
    name="DFlashKimiK25DraftModel",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["nvidia/Kimi-K2.5-NVFP4"],
    default_encoding="bfloat16",
    supported_encodings={"bfloat16"},
    # Sharded over the target's devices, as the target is.
    multi_gpu_supported=True,
    pipeline_model=None,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    default_weights_format=WeightsFormat.safetensors,
    config=DFlashKimiK25DraftArchConfig,
    supports_device_graph_capture=False,
    checkpoint_draft_width=dflash_draft_width,
    input_modalities={InputModality.TEXT, InputModality.IMAGE},
)
