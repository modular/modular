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

from .model import ReplitModel

replit_arch = SupportedArchitecture(
    name="MPTForCausalLM",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=[
        "modularai/replit-code-1.5",
    ],
    default_encoding=SupportedEncoding.bfloat16,
    supported_encodings={
        SupportedEncoding.float32: [KVCacheStrategy.CONTINUOUS],
        SupportedEncoding.bfloat16: [KVCacheStrategy.CONTINUOUS],
    },
    pipeline_model=ReplitModel,
    tokenizer=TextTokenizer,
    default_weights_format=WeightsFormat.gguf,
)
