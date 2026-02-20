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
from max.interfaces import PipelineTask
from max.pipelines.core import TextContext

# TODO(bduke): Replace with actual Llama4 model once implemented.
from max.pipelines.lib import (
    SupportedArchitecture,
    TextTokenizer,
)

from . import weight_adapters
from .model import Llama4Model
from .model_config import Llama4Config

llama4_arch = SupportedArchitecture(
    name="Llama4ForConditionalGeneration_Legacy",
    example_repo_ids=[
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "meta-llama/Llama-4-Scout-17B-16E",
        "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
        "meta-llama/Llama-4-Maverick-17B-128E",
    ],
    default_encoding="bfloat16",
    supported_encodings={
        "bfloat16": ["paged"],
    },
    pipeline_model=Llama4Model,
    task=PipelineTask.TEXT_GENERATION,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=True,
    # NOTE: Llama 4 interleaves RoPE and NoPE (no positional encodings).
    rope_type="normal",
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
    config=Llama4Config,
)
