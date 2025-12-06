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
    RopeType,
    SupportedArchitecture,
    SupportedEncoding,
    TextTokenizer,
)

from .model import ZImageModel
from max.pipelines.architectures.qwen3 import qwen3_arch
from .nn.transformer_z_image import ZImageTransformer2DModel
from .nn.autoencoderkl import AutoencoderKL
from .scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from .weight_adapters import convert_z_image_model_state_dict

z_image_arch = SupportedArchitecture(
    name="ZImagePipeline",
    task=PipelineTask.IMAGE_GENERATION,
    example_repo_ids=[
        "Tongyi-MAI/Z-Image-Base",
        "Tongyi-MAI/Z-Image-Turbo",
    ],
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=True,
    default_encoding=SupportedEncoding.bfloat16,
    supported_encodings={
        SupportedEncoding.float32: None,
        SupportedEncoding.bfloat16: None,
    },
    weight_adapters={
        WeightsFormat.safetensors: convert_z_image_model_state_dict,
    },
    pipeline_model=ZImageModel,
    scheduler=FlowMatchEulerDiscreteScheduler,
    vae=AutoencoderKL,
    text_encoder=qwen3_arch,
    tokenizer=TextTokenizer,
    transformer=ZImageTransformer2DModel,
    rope_type=RopeType.normal,
    required_arguments={
        "enable_chunked_prefill": False,
    },
)
