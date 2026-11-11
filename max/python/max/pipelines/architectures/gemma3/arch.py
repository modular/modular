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

"""Register the Gemma 3 text-only architecture."""

from max.graph.weights import WeightsFormat
from max.pipelines.context import TextContext
from max.pipelines.lib import GraphRole, SupportedArchitecture, TextTokenizer
from max.pipelines.modeling.types import PipelineTask

from .gemma3 import Gemma3
from .model_config import Gemma3Config
from .weight_adapters import gemma3_weight_rules

gemma3_arch = SupportedArchitecture(
    name="Gemma3ForCausalLM",
    example_repo_ids=[
        # it = Instruction tuned (recommended).
        # pt = Pre-trained.
        "google/gemma-3-1b-it",
        "google/gemma-3-1b-pt",
        # We have a different architecture for >=4B models. See gemma3multimodal
        # for more information.
    ],
    default_encoding=Gemma3Config.DEFAULT_ENCODING,
    supported_encodings=Gemma3Config.SUPPORTED_ENCODINGS,
    modules={GraphRole.DECODER: Gemma3},
    task=PipelineTask.TEXT_GENERATION,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=True,
    weight_rules=gemma3_weight_rules,
    use_weight_rules=True,
    config=Gemma3Config,
    supports_overlap_scheduler=False,
    supports_device_graph_capture=False,
)
