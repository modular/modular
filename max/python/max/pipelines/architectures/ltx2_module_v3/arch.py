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
from max.interfaces import PipelineTask, PixelGenerationContext
from max.nn.kv_cache import KVCacheStrategy
from max.pipelines.architectures.gemma3.gemma3 import Gemma3
from max.pipelines.architectures.llama3 import weight_adapters
from max.pipelines.lib import (
    RopeType,
    SupportedArchitecture,
    SupportedEncoding,
    TextTokenizer,
)

from ..z_image_module_v3.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from .model import LTX2Model
from .nn.autoencoder_kl_ltx2 import AutoencoderKLLTX2Video
from .nn.autoencoder_kl_ltx2_audio import AutoencoderKLLTX2Audio
from .nn.transformer_ltx2 import LTX2Transformer3DModel

ltx2_module_v3_arch = SupportedArchitecture(
    name="LTX2Pipeline",
    task=PipelineTask.PIXEL_GENERATION,
    example_repo_ids=["Lightricks/LTX-2"],
    default_weights_format=WeightsFormat.safetensors,
    default_encoding=SupportedEncoding.bfloat16,
    supported_encodings={
        SupportedEncoding.bfloat16: [KVCacheStrategy.MODEL_DEFAULT]
    },
    pipeline_model=LTX2Model,
    scheduler=FlowMatchEulerDiscreteScheduler,
    vae=AutoencoderKLLTX2Video,
    vae_audio=AutoencoderKLLTX2Audio,
    text_encoder=Gemma3,
    tokenizer=TextTokenizer,
    transformer=LTX2Transformer3DModel,
    context_type=PixelGenerationContext,
    rope_type=RopeType.normal,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict
    },
)
