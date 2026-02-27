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

from max.graph.weights import WeightsFormat
from max.interfaces import PipelineTask
from max.pipelines.core import TextContext
from max.pipelines.lib import (
    SupportedArchitecture,
    TextTokenizer,
)
from transformers import AutoConfig, DeepseekV3Config

from . import weight_adapters
from .model import DeepseekV3_2Model
from .model_config import DeepseekV3_2Config

deepseekV3_2_arch = SupportedArchitecture(
    name="DeepseekV32ForCausalLM",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=[
        "deepseek-ai/DeepSeek-V3.2",
        "deepseek-ai/DeepSeek-V3.2-Exp",
    ],
    default_encoding="float8_e4m3fn",
    supported_encodings={
        "float8_e4m3fn": ["paged"],
    },
    multi_gpu_supported=True,
    pipeline_model=DeepseekV3_2Model,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    default_weights_format=WeightsFormat.safetensors,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
    supports_empty_batches=True,
    requires_max_batch_context_length=True,
    config=DeepseekV3_2Config,
)


class DeepseekV32HFConfig(DeepseekV3Config):
    """HuggingFace configuration class for DeepSeek-V3.2 models.

    The ``deepseek_v32`` model type is not natively registered in transformers.
    This subclass of ``DeepseekV3Config`` adds the V3.2-specific fields for
    sparse attention (indexer) and registers itself so that
    ``AutoConfig.from_pretrained`` can load DeepSeek-V3.2 repos.
    """

    model_type = "deepseek_v32"

    def __init__(
        self,
        index_head_dim: int = 128,
        index_n_heads: int = 64,
        index_topk: int = 2048,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads
        self.index_topk = index_topk


# Register the config with AutoConfig if not already registered.
# This allows AutoConfig.from_pretrained() to work with deepseek_v32 models.
try:
    AutoConfig.register("deepseek_v32", DeepseekV32HFConfig)
except ValueError:
    # Already registered, which is fine.
    pass
