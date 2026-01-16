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
from max.interfaces import BaseContext, PipelineTask
from max.pipelines.lib import (
    SupportedArchitecture,
    SupportedEncoding,
    TextTokenizer,
)

from .pipeline_flux import FluxPipeline

# TODO(minkyu): revisit default_encoding, supported_encodings, tokenizer.
flux1_arch = SupportedArchitecture(
    name="FluxPipeline",
    task=PipelineTask.IMAGE_GENERATION,
    default_encoding=SupportedEncoding.bfloat16,
    supported_encodings={SupportedEncoding.bfloat16: []},
    example_repo_ids=[
        "black-forest-labs/FLUX.1-dev",
        "black-forest-labs/FLUX.1-schnell",
    ],
    pipeline_model=FluxPipeline,
    tokenizer=TextTokenizer,
    context_type=BaseContext,
    default_weights_format=WeightsFormat.safetensors,
)
