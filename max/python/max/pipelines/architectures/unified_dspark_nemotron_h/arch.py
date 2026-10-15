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

from dataclasses import replace

from max.graph.weights import WeightsFormat
from max.pipelines.context import TextContext
from max.pipelines.lib import SupportedArchitecture, TextTokenizer
from max.pipelines.modeling.types import PipelineTask

from ..nemotron_h_modulev3.arch import nemotron_h_modulev3_arch
from .batch_processor import UnifiedDSparkNemotronHBatchProcessor
from .memory_planner import UnifiedDSparkNemotronHMemoryPlanner
from .model import UnifiedDSparkNemotronHModel
from .model_config import (
    Qwen3DSparkDraftArchConfig,
    UnifiedDSparkNemotronHConfig,
    dspark_nemotron_draft_width,
)

# Everything but the model matches the Nemotron-H ModuleV3 architecture: the
# same tokenizer, parsers, weight adapter and context. The fused graph is
# registered under the ModuleV3 name only; a lookup without the ModuleV3
# preference falls back to it.
unified_dspark_nemotron_h_arch = replace(
    nemotron_h_modulev3_arch,
    name="UnifiedDSparkNemotronHForCausalLM_ModuleV3",
    example_repo_ids=[
        "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
    ],
    pipeline_model=UnifiedDSparkNemotronHModel,
    batching=UnifiedDSparkNemotronHBatchProcessor,
    config=UnifiedDSparkNemotronHConfig,
    memory_planner=UnifiedDSparkNemotronHMemoryPlanner,
    checkpoint_draft_width=dspark_nemotron_draft_width,
    # A mixed prefill and decode batch verifies nothing: its decode rows
    # advance draft-less, which the Mamba commit of a step with no drafts
    # handles, whereas a per-row mixed verify would need every prefill row's
    # whole chunk captured for replay.
    supports_spec_decode_mixed_batches=False,
    # The Mamba rollback is shape-stable (the replay plan never sizes a gather
    # by the accepted count), and the block draft's dispatch metadata is
    # captured as for the other block drafts.
    supports_device_graph_capture=True,
)

# The DSpark draft checkpoint declares architectures ["Qwen3DSparkModel"]. It
# is only served as a --draft-model: the config resolution rewrites the
# (NemotronHForCausalLM target, Qwen3DSparkModel draft) pair to the unified
# architecture above, so this registration exists for draft-side name lookup
# and the max-sequence-length clamp. The pipeline_model / tokenizer /
# context_type placeholders are never constructed for a draft.
qwen3_dspark_draft_arch = SupportedArchitecture(
    name="Qwen3DSparkModel",
    example_repo_ids=[
        "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark",
    ],
    default_encoding="bfloat16",
    # The released draft is NVFP4 weight-only, which the checkpoint's
    # quantization config resolves to; a BF16 drafter resolves to bfloat16.
    supported_encodings={"bfloat16", "float4_e2m1fnx2"},
    pipeline_model=UnifiedDSparkNemotronHModel,
    context_type=TextContext,
    tokenizer=TextTokenizer,
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=True,
    task=PipelineTask.TEXT_GENERATION,
    config=Qwen3DSparkDraftArchConfig,
    checkpoint_draft_width=dspark_nemotron_draft_width,
)
