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
"""Registers Hy4 and Hy4 VL with the ModuleV3 registry."""

from dataclasses import replace

from max.graph.weights import WeightsFormat
from max.pipelines.context import TextContext
from max.pipelines.lib import GraphRole, SupportedArchitecture
from max.pipelines.modeling.types import PipelineTask

from .decoder import HYV4Decoder
from .model_config import HYV4Config, HYV4VLConfig
from .reasoning import HYV4ReasoningParser  # noqa: F401  registers "hy_v4"
from .tokenizer import HYV4Tokenizer
from .tool_parser import HYV4ToolParser  # noqa: F401  registers "hy_v4"
from .weight_adapters import hy_v4_vl_weight_rules, hy_v4_weight_rules

hy_v4_arch = SupportedArchitecture(
    name="HYV4ForCausalLM_ModuleV3",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["tencent/Hy4-preview-FP8"],
    default_encoding=HYV4Config.DEFAULT_ENCODING,
    supported_encodings=HYV4Config.SUPPORTED_ENCODINGS,
    modules={GraphRole.DECODER: HYV4Decoder},
    tokenizer=HYV4Tokenizer,
    context_type=TextContext,
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=True,
    requires_max_batch_context_length=True,
    # The DSA indexer picks its scorer and top-k arms from the host cache
    # bound, so a capture keys and pins the bound's power-of-two bucket.
    capture_cache_bound="bucketed",
    # The indexer kernels size their own launches.
    unread_dispatch_caches=("indexer",),
    checkpoint_drafts={"mtp": "MTPHYV4DraftModel"},
    weight_rules=hy_v4_weight_rules,
    use_weight_rules=True,
    config=HYV4Config,
    tool_parser="hy_v4",
    reasoning_parser="hy_v4",
    mach_only=True,
)

# Hy4-FP8 ships no MTP layers; the text backbone reuses Hy4's decoder.
hy_v4_vl_arch = replace(
    hy_v4_arch,
    name="HYV4VLForConditionalGeneration_ModuleV3",
    example_repo_ids=["tencent/Hy4-FP8"],
    config=HYV4VLConfig,
    weight_rules=hy_v4_vl_weight_rules,
    checkpoint_drafts={},
    # replace() copies the pipeline model derived from HYV4Config; None
    # re-derives it from HYV4VLConfig.
    pipeline_model=None,
)
