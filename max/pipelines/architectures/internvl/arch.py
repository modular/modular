# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
from max.nn.kv_cache import KVCacheStrategy
from max.pipelines.lib import SupportedArchitecture, SupportedEncoding

from .model import InternVLModel
from .tokenizer import InternVLTokenizer

internvl_arch = SupportedArchitecture(
    name="InternVLChatModel",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["OpenGVLab/InternVL3-8B-Instruct"],
    default_encoding=SupportedEncoding.bfloat16,
    supported_encodings={SupportedEncoding.bfloat16: [KVCacheStrategy.PAGED]},
    pipeline_model=InternVLModel,
    tokenizer=InternVLTokenizer,
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=True,
    prefix_caching_supported=False,
)
