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
)

from .model import BertPipelineModel
from .tokenizer import BertTextTokenizer

bert_arch = SupportedArchitecture(
    name="BertForMaskedLM",
    task=PipelineTask.EMBEDDINGS_GENERATION,
    example_repo_ids=[
        "bert-base-uncased",
        "bert-large-uncased",
    ],
    default_encoding=SupportedEncoding.bfloat16,
    supported_encodings={
        SupportedEncoding.float32: [],
        SupportedEncoding.bfloat16: [],
    },
    pipeline_model=BertPipelineModel,
    tokenizer=BertTextTokenizer,
    default_weights_format=WeightsFormat.safetensors,
)

# Support for BertModel (used by sentence-transformers)
bert_model_arch = SupportedArchitecture(
    name="BertModel",
    task=PipelineTask.EMBEDDINGS_GENERATION,
    example_repo_ids=[
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2",
    ],
    default_encoding=SupportedEncoding.bfloat16,
    supported_encodings={
        SupportedEncoding.float32: [],
        SupportedEncoding.bfloat16: [],
    },
    pipeline_model=BertPipelineModel,
    tokenizer=BertTextTokenizer,  # Use BertTextTokenizer to handle missing EOS
    default_weights_format=WeightsFormat.safetensors,
)
