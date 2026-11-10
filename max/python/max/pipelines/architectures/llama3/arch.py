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
"""Registers Llama3 with the pipeline registry."""

import re

from max.experimental.cascade.pipelines.common_textgen import (
    CommonTextGenPipeline,
)
from max.graph.weights import WeightsFormat
from max.pipelines.context import TextContext
from max.pipelines.kv_cache.memory_planner import PagedMemoryPlanner
from max.pipelines.lib import GraphRole, SupportedArchitecture, TextTokenizer
from max.pipelines.modeling.types import PipelineTask
from max.pipelines.weights.hf_utils import HuggingFaceRepo

from . import weight_adapters
from .decoder import Llama3DecoderConfig, Llama3TargetDecoder
from .model_config import Llama3Config


def llama3_tokenizer_family(
    repo: HuggingFaceRepo,
) -> tuple[str, str | None] | None:
    """Returns the tokenizer family of a Llama checkpoint.

    Args:
        repo: The checkpoint.

    Returns:
        The Llama 3 family and its Llama 3.1 and 3.3 or Llama 3.2 flavor, or
        ``None`` for another checkpoint the architecture serves.
    """
    name = repo.repo_id.lower().rsplit("/", 1)[-1]
    if not name.startswith(("llama-3", "meta-llama-3")) or "vision" in name:
        return None
    if re.search(r"(?<![a-z0-9])3[.-]2(?![a-z0-9])", name):
        return ("llama3", "lite")
    if re.search(r"(?<![a-z0-9])3[.-][13](?![a-z0-9])", name):
        return ("llama3", "standard")
    return None


llama_arch = SupportedArchitecture(
    name="LlamaForCausalLM",
    example_repo_ids=[
        "meta-llama/Llama-3.1-8B-Instruct",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "meta-llama/Llama-Guard-3-8B",
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "deepseek-ai/deepseek-coder-6.7b-instruct",
        "modularai/Llama-3.1-8B-Instruct-GGUF",
    ],
    # The models table generator reads these off the class that defines them.
    default_encoding=Llama3Config.DEFAULT_ENCODING,
    supported_encodings=Llama3Config.SUPPORTED_ENCODINGS,
    modules={GraphRole.DECODER: Llama3TargetDecoder},
    tokenizer=TextTokenizer,
    tokenizer_family=llama3_tokenizer_family,
    context_type=TextContext,
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=True,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
        WeightsFormat.gguf: weight_adapters.convert_gguf_state_dict,
    },
    task=PipelineTask.TEXT_GENERATION,
    config=Llama3DecoderConfig,
    memory_planner=PagedMemoryPlanner,
    cascade_pipeline_factory=CommonTextGenPipeline,
    # An EAGLE checkpoint names the Llama architecture of its one decoder layer.
    draft_architectures={
        "eagle": {"LlamaForCausalLM": "LlamaForCausalLMEagle"}
    },
    supports_spec_decode_mixed_batches=True,
)
