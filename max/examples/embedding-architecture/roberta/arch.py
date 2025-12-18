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
from max.pipelines.lib import (
    SupportedArchitecture,
    SupportedEncoding,
    TextTokenizer,
)

from .model import RobertaPipelineModel

# RoBERTa models with proper EOS tokens
roberta_arch = SupportedArchitecture(
    name="RobertaForMaskedLM",
    task=PipelineTask.EMBEDDINGS_GENERATION,
    example_repo_ids=[
        "roberta-base",
        "roberta-large",
        "FacebookAI/roberta-base",
        "FacebookAI/roberta-large",
    ],
    default_encoding=SupportedEncoding.bfloat16,
    supported_encodings={
        SupportedEncoding.float32: [],
        SupportedEncoding.bfloat16: [],
    },
    pipeline_model=RobertaPipelineModel,
    tokenizer=TextTokenizer,  # RoBERTa has proper EOS tokens
    default_weights_format=WeightsFormat.safetensors,
)

# DistilRoBERTa models
distilroberta_arch = SupportedArchitecture(
    name="RobertaForMaskedLM",  # DistilRoBERTa uses same architecture name
    task=PipelineTask.EMBEDDINGS_GENERATION,
    example_repo_ids=[
        "distilroberta-base",
        "sentence-transformers/all-distilroberta-v1",
    ],
    default_encoding=SupportedEncoding.bfloat16,
    supported_encodings={
        SupportedEncoding.float32: [],
        SupportedEncoding.bfloat16: [],
    },
    pipeline_model=RobertaPipelineModel,
    tokenizer=TextTokenizer,
    default_weights_format=WeightsFormat.safetensors,
)
