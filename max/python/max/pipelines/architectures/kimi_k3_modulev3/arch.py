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
The tower builds for a speculative pairing through the framework's
registered builder, on a config the pairing stamped with the draft's
``target_layer_ids``, so it reads the attention-residual mixture the
drafter was trained against (TP only, the config refusing expert
parallelism there); the fused pairing itself lives in the DSpark drafter's
package (``kimi_k3_modulev3.dspark``).
"""

from __future__ import annotations

from max.graph.weights import WeightsFormat
from max.pipelines.context import TextContext
from max.pipelines.lib import GraphRole, SupportedArchitecture, TextTokenizer
from max.pipelines.modeling.types import PipelineTask

from .kimi_k3 import KimiK3Decoder
from .memory_planner import KimiK3ModuleV3MemoryPlanner
from .model_config import KimiK3Config
from .weight_adapters import convert_kimi_k3_state_dict

__all__ = ["kimi_k3_modulev3_arch"]

kimi_k3_modulev3_arch = SupportedArchitecture(
    name="KimiK3ForConditionalGeneration_ModuleV3",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["inference-optimization/Kimi-K3-0.40B"],
    default_weights_format=WeightsFormat.safetensors,
    default_encoding=KimiK3Config.DEFAULT_ENCODING,
    supported_encodings=KimiK3Config.SUPPORTED_ENCODINGS,
    modules={GraphRole.DECODER: KimiK3Decoder},
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
    # The fused graph the `Inferact/Kimi-K3-DSpark` drafter builds with this
    # target. Named apart from the graph API path's `unified_dspark_kimi_k3`
    # because the two ABIs share nothing but arity: the ModuleV3 graph ends
    # its declared inputs with the PR7 extent pair and takes its traced signal
    # buffers after it (mach/docs/models/kimi-k3-inferact-dspark.md). The
    # exporter selects the fused submodel by this name.
    spec_decode_graph_names={"dspark": "unified_dspark_kimi_k3_modulev3"},
)
