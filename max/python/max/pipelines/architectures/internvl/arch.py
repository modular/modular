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
from max.pipelines.core import TextAndVisionContext
from max.pipelines.lib import SupportedArchitecture

from .model import InternVLModel
from .model_config import InternVLConfig
from .tokenizer import InternVLTokenizer

internvl_arch = SupportedArchitecture(
    name="InternVLChatModel_Legacy",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["OpenGVLab/InternVL3-8B-Instruct"],
    default_encoding="bfloat16",
    supported_encodings={"bfloat16": ["paged"]},
    pipeline_model=InternVLModel,
    tokenizer=InternVLTokenizer,
    context_type=TextAndVisionContext,
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=True,
    required_arguments={
        "enable_prefix_caching": False,
        "enable_chunked_prefill": False,
    },
    config=InternVLConfig,
)
