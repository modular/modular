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
from max.pipelines.lib import RopeType, SupportedArchitecture, SupportedEncoding

from .model import QwenImageEdit2511Model
from ..qwen2_5vl import qwen2_5_vl_arch
from ..qwen2_5vl.tokenizer import Qwen2_5VLTokenizer
from .nn.transformer_qwenimage import QwenImageTransformer2DModel
from .nn.autoencoderkl_qwenimage import AutoencoderKLQwenImage
from .scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from .weight_adapters import convert_qwen_image_edit_2511_model_state_dict

qwen_image_edit_2511_arch = SupportedArchitecture(
    name="QwenImageEdit2511Pipeline",
    task=PipelineTask.IMAGE_GENERATION,
    example_repo_ids=[
        "Qwen/Qwen-Image-Edit-2511",
    ],
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=True,
    default_encoding=SupportedEncoding.bfloat16,
    supported_encodings={
        SupportedEncoding.float32: None,
        SupportedEncoding.bfloat16: None,
    },
    weight_adapters={
        WeightsFormat.safetensors: convert_qwen_image_edit_2511_model_state_dict,
    },
    pipeline_model=QwenImageEdit2511Model,
    scheduler=FlowMatchEulerDiscreteScheduler,
    vae=AutoencoderKLQwenImage,
    text_encoder=qwen2_5_vl_arch,
    tokenizer=Qwen2_5VLTokenizer,
    transformer=QwenImageTransformer2DModel,
    rope_type=RopeType.normal,
    required_arguments={
        "enable_chunked_prefill": False,
    },
)
