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
"""Kimi K3 on ModuleV3.

Selected over the Graph-API architecture (``kimi_k3.arch.kimi_k3_arch``) by
``prefer_module_v3``; both read the same checkpoint and the same cache tree.

TP, EP and MXFP4 are all wired (see ``layers/ep.py``, ``layers/moe_latent.py``
and ``weight_adapters.py``); EP and MXFP4 are mutually exclusive, matching
``moonshotai/Kimi-K3`` (MXFP4) vs ``moonshotai/Kimi-K3-ep`` (EP, bf16).
Speculative decoding is not ported.
"""

from __future__ import annotations

from max.graph.weights import WeightsFormat
from max.pipelines.context import TextContext
from max.pipelines.lib import SupportedArchitecture, TextTokenizer
from max.pipelines.modeling.types import PipelineTask

from .batch_processor import KimiK3ModuleV3BatchProcessor
from .memory_planner import KimiK3ModuleV3MemoryPlanner
from .model_config import KimiK3Config
from .pipeline_model import KimiK3PipelineModel
from .weight_adapters import convert_kimi_k3_state_dict

__all__ = ["kimi_k3_modulev3_arch"]

kimi_k3_modulev3_arch = SupportedArchitecture(
    name="KimiK3ForConditionalGeneration_ModuleV3",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["inference-optimization/Kimi-K3-0.40B"],
    default_weights_format=WeightsFormat.safetensors,
    default_encoding=KimiK3Config.DEFAULT_ENCODING,
    supported_encodings=KimiK3Config.SUPPORTED_ENCODINGS,
    pipeline_model=KimiK3PipelineModel,
    batching=KimiK3ModuleV3BatchProcessor,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    weight_adapters={
        WeightsFormat.safetensors: convert_kimi_k3_state_dict,
    },
    required_arguments={
        # The KDA recurrent state is not reconstructible from a page table,
        # so a prefix hit would resume from the wrong state.
        "enable_prefix_caching": False,
    },
    config=KimiK3Config,
    memory_planner=KimiK3ModuleV3MemoryPlanner,
    multi_gpu_supported=True,
    requires_max_batch_context_length=True,
)
