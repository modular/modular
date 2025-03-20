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
from max.pipelines import (
    PipelineTask,
    SupportedArchitecture,
    SupportedEncoding,
    TextAndVisionTokenizer,
)
from max.pipelines.kv_cache import KVCacheStrategy

from .pixtral import PixtralModel

pixtral_arch = SupportedArchitecture(
    name="LlavaForConditionalGeneration",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["mistral-community/pixtral-12b"],
    default_encoding=SupportedEncoding.bfloat16,
    supported_encodings={
        SupportedEncoding.bfloat16: [
            KVCacheStrategy.PAGED,
            KVCacheStrategy.CONTINUOUS,
        ],
    },
    pipeline_model=PixtralModel,
    tokenizer=TextAndVisionTokenizer,
    default_weights_format=WeightsFormat.safetensors,
)
