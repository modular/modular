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
from max.pipelines.architectures.llama3 import weight_adapters
from max.pipelines.architectures.qwen3.qwen3 import Qwen3
from max.pipelines.lib import (
    RopeType,
    SupportedArchitecture,
    SupportedEncoding,
    TextTokenizer,
)

from .model import ZImageModel
from .nn.autoencoder_kl import AutoencoderKL
from .nn.transformer_z_image import ZImageTransformer2DModel
from .scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)

z_image_module_v3_arch = SupportedArchitecture(
    name="ZImagePipeline",
    task=PipelineTask.PIXEL_GENERATION,
    example_repo_ids=["Tongyi-MAI/Z-Image-Turbo"],
    default_weights_format=WeightsFormat.safetensors,
    default_encoding=SupportedEncoding.bfloat16,
    supported_encodings={
        SupportedEncoding.bfloat16: [KVCacheStrategy.MODEL_DEFAULT]
    },  # No KV Caching for image-gen pipelines.
    pipeline_model=ZImageModel,
    scheduler=FlowMatchEulerDiscreteScheduler,
    vae=AutoencoderKL,
    text_encoder=Qwen3,
    tokenizer=TextTokenizer,
    transformer=ZImageTransformer2DModel,
    context_type=PixelGenerationContext,
    rope_type=RopeType.normal,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict
    },
)
