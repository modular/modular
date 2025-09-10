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

from .model import Idefics3Model
from .tokenizer import Idefics3Tokenizer

idefics3_arch = SupportedArchitecture(
    name="Idefics3ForConditionalGeneration",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["HuggingFaceM4/Idefics3-8B-Llama3"],
    default_encoding=SupportedEncoding.bfloat16,
    supported_encodings={
        SupportedEncoding.bfloat16: [KVCacheStrategy.PAGED],
    },
    pipeline_model=Idefics3Model,
    tokenizer=Idefics3Tokenizer,
    default_weights_format=WeightsFormat.safetensors,
    prefix_caching_supported=False,
)
