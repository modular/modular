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
"""Nemotron-H architecture registration."""

from __future__ import annotations

from max.graph.weights import WeightsFormat
from max.pipelines.context import TextContext
from max.pipelines.lib import SupportedArchitecture, TextTokenizer
from max.pipelines.modeling.types import PipelineTask

from .model import NemotronHModel
from .model_config import NemotronHConfig
from .weight_adapters import convert_nemotron_h_state_dict

nemotron_h_arch = SupportedArchitecture(
    name="NemotronHForCausalLM",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8"],
    default_weights_format=WeightsFormat.safetensors,
    default_encoding="bfloat16",
    # modelopt per-tensor static FP8 on the (non-excluded) mamba in/out_proj and
    # MLP up/down projections; attention, conv1d, norms, lm_head stay bf16.
    supported_encodings={"bfloat16", "float8_e4m3fn"},
    pipeline_model=NemotronHModel,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    # NoPE: attention adds no rotary embedding (position flows through the SSM).
    rope_type="none",
    weight_adapters={
        WeightsFormat.safetensors: convert_nemotron_h_state_dict,
    },
    # SSM recurrent state is not reconstructable from a token prefix, so prefix
    # caching must be disabled.
    required_arguments={"enable_prefix_caching": False},
    config=NemotronHConfig,
    multi_gpu_supported=False,
    # SSM functional state scatter / slot-indexed conv pools are not
    # device-graph-capture safe.
    supports_device_graph_capture=False,
)
