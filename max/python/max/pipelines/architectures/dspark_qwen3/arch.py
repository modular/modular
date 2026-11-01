# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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
"""The ``Qwen3DSparkModel`` draft architecture."""

from max.graph.weights import WeightsFormat
from max.pipelines.context import TextContext
from max.pipelines.lib import SupportedArchitecture, TextTokenizer
from max.pipelines.modeling.types import PipelineTask

from .model_config import (
    Qwen3DSparkDraftArchConfig,
    dspark_nemotron_draft_width,
)

# The DSpark draft checkpoint declares architectures ["Qwen3DSparkModel"]. It
# is only served as a --draft-model beside a Nemotron-H target, so its
# tokenizer and context type are never used.
qwen3_dspark_draft_arch = SupportedArchitecture(
    name="Qwen3DSparkModel",
    example_repo_ids=[
        "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark",
    ],
    default_encoding="bfloat16",
    # The released draft is NVFP4 weight-only, which the checkpoint's
    # quantization config resolves to; a BF16 drafter resolves to bfloat16.
    supported_encodings={"bfloat16", "float4_e2m1fnx2"},
    pipeline_model=None,
    context_type=TextContext,
    tokenizer=TextTokenizer,
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=True,
    task=PipelineTask.TEXT_GENERATION,
    config=Qwen3DSparkDraftArchConfig,
    checkpoint_draft_width=dspark_nemotron_draft_width,
)
