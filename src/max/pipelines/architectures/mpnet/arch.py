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

from .model import MPNetPipelineModel

mpnet_arch = SupportedArchitecture(
    name="MPNetForMaskedLM",
    task=PipelineTask.EMBEDDINGS_GENERATION,
    example_repo_ids=[
        "sentence-transformers/all-mpnet-base-v2",
    ],
    default_encoding=SupportedEncoding.bfloat16,
    supported_encodings={
        SupportedEncoding.float32: [],
        SupportedEncoding.bfloat16: [],
    },
    pipeline_model=MPNetPipelineModel,
    tokenizer=TextTokenizer,
    default_weights_format=WeightsFormat.safetensors,
)
