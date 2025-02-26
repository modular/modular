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

from max.pipelines import (
    PipelineTask,
    SupportedArchitecture,
    SupportedEncoding,
    TextTokenizer,
    WeightsFormat,
)
from max.pipelines.kv_cache import KVCacheStrategy

from .model import Qwen2Model

qwen2_arch = SupportedArchitecture(
    name="Qwen2ForCausalLM",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=[
        "Qwen/Qwen2.5-7B-Instruct",
    ],
    default_weights_format=WeightsFormat.safetensors,
    default_encoding=SupportedEncoding.bfloat16,
    supported_encodings={
        SupportedEncoding.float32: [
            KVCacheStrategy.PAGED,
            KVCacheStrategy.CONTINUOUS,
            KVCacheStrategy.NAIVE,
        ],
        SupportedEncoding.bfloat16: [
            KVCacheStrategy.PAGED,
            KVCacheStrategy.CONTINUOUS,
            KVCacheStrategy.NAIVE,
        ],
    },
    pipeline_model=Qwen2Model,
    tokenizer=TextTokenizer,
)
