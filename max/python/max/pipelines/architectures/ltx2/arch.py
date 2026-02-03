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
from max.interfaces import PipelineTask, PixelContext
from max.pipelines.lib import (
    PixelGenerationTokenizer,
    SupportedArchitecture,
    SupportedEncoding,
)

from .pipeline_ltx2 import LTX2Pipeline

ltx2_arch = SupportedArchitecture(
    name="LTX2Pipeline",
    task=PipelineTask.PIXEL_GENERATION,
    default_encoding=SupportedEncoding.bfloat16,
    supported_encodings={SupportedEncoding.bfloat16: []},
    example_repo_ids=["Lightricks/LTX-2"],
    pipeline_model=LTX2Pipeline,  # type: ignore[arg-type]
    context_type=PixelContext,
    default_weights_format=WeightsFormat.safetensors,
    tokenizer=PixelGenerationTokenizer,
)
