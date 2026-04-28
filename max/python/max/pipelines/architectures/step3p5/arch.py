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
from transformers import AutoConfig, PretrainedConfig

from .model import Step3p5Model
from .model_config import Step3p5Config
from .weight_adapters import convert_step3p5_state_dict


# Register custom config since "step3p5" is not in the transformers library.
class Step3p5PretrainedConfig(PretrainedConfig):
    """Custom PretrainedConfig for Step-3.5 so AutoConfig.from_pretrained() works.

    This is the primary location for mapping Step-3.5 field names to the
    standard HuggingFace fields that Llama3Config expects.  A subset of these
    aliases is also applied in Step3p5Config._ensure_hf_config_aliases() as a
    fallback when trust_remote_code=True loads the repo's own config class
    instead of this one.
    """

    model_type = "step3p5"

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            if not hasattr(self, k):
                setattr(self, k, v)
        # Standard field aliases that Llama3Config reads.
        if not hasattr(self, "num_key_value_heads"):
            self.num_key_value_heads = getattr(self, "num_attention_groups", 8)
        if not hasattr(self, "rms_norm_eps"):
            self.rms_norm_eps = 1e-5
        if not hasattr(self, "rope_scaling"):
            self.rope_scaling = None
        if not hasattr(self, "hidden_act"):
            self.hidden_act = "silu"
        # rope_theta may be a per-layer list; preserve it and set scalar.
        rope_theta = getattr(self, "rope_theta", 10000.0)
        if isinstance(rope_theta, list):
            self.per_layer_rope_theta = rope_theta
            self.rope_theta = rope_theta[0] if rope_theta else 10000.0
        elif not hasattr(self, "per_layer_rope_theta"):
            self.per_layer_rope_theta = []


try:
    AutoConfig.register("step3p5", Step3p5PretrainedConfig)
except ValueError:
    pass  # Already registered


step3p5_arch = SupportedArchitecture(
    name="Step3p5ForCausalLM",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["stepfun-ai/Step-3.5-Flash"],
    default_weights_format=WeightsFormat.safetensors,
    default_encoding="bfloat16",
    supported_encodings={"bfloat16"},
    pipeline_model=Step3p5Model,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    rope_type="normal",
    weight_adapters={
        WeightsFormat.safetensors: convert_step3p5_state_dict,
    },
    config=Step3p5Config,
    multi_gpu_supported=True,
)
