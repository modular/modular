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
from max.pipelines.lib import GraphRole, SupportedArchitecture
from max.pipelines.lib.tokenizer import ReasoningTextTokenizer
from max.pipelines.modeling.types import PipelineTask

# Nemotron-3's chat template is Qwen-format: it pre-fills ``<think>\n`` in the
# generation prompt (implicit reasoning open, explicit ``</think>`` close),
# backfills ``<think></think>`` into prior assistant turns, and renders tool
# calls as ``<tool_call>/<function=...>/<parameter=...>`` blocks. Built-in
# architectures load lazily, so import the qwen3_5 parser modules here to
# register the parsers named below.
from ..qwen3_5.reasoning import (
    Qwen3_5ReasoningParser,  # noqa: F401  registers "qwen3_5"
)
from ..qwen3_5.tool_parser import (
    Qwen3_5ToolParser,  # noqa: F401  registers "qwen3_5"
)
from .model_config import NemotronHConfig
from .nemotron_h import NemotronH
from .weight_adapters import convert_nemotron_h_state_dict

nemotron_h_arch = SupportedArchitecture(
    name="NemotronHForCausalLM",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=[
        "nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8",
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
    ],
    default_weights_format=WeightsFormat.safetensors,
    default_encoding=NemotronHConfig.DEFAULT_ENCODING,
    # modelopt per-tensor static FP8 on the (non-excluded) mamba in/out_proj and
    # MLP up/down projections; attention, conv1d, norms, lm_head stay bf16.
    supported_encodings=NemotronHConfig.SUPPORTED_ENCODINGS,
    modules={GraphRole.DECODER: NemotronH},
    tokenizer=ReasoningTextTokenizer,
    context_type=TextContext,
    # NoPE: attention adds no rotary embedding (position flows through the SSM).
    weight_adapters={
        WeightsFormat.safetensors: convert_nemotron_h_state_dict,
    },
    checkpoints_recurrent_state=True,
    config=NemotronHConfig,
    multi_gpu_supported=False,
    # Reasoning opens implicitly (template pre-fills ``<think>``) and closes
    # at ``</think>``; without a default parser the CoT and the raw
    # ``</think>`` delimiter leak into OpenAI ``message.content``. The
    # qwen3_5 parsers match Nemotron-3's Qwen-format template exactly.
    reasoning_parser="qwen3_5",
    tool_parser="qwen3_5",
    supports_device_graph_capture=True,
)
