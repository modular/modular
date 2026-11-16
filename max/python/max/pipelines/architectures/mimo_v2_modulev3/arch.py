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
"""Registers MiMo-V2 with the ModuleV3 registry."""

from pathlib import Path

from max.graph.weights import WeightsFormat
from max.pipelines.context import TextContext
from max.pipelines.lib import SupportedArchitecture, TextTokenizer
from max.pipelines.lib.graph_specs import GraphRole
from max.pipelines.modeling.types import PipelineTask

from .mimo_v2 import MiMoV2Decoder
from .model_config import MiMoV2Config
from .weight_rules import mimo_v2_weight_rules

# The tokenizer, parsers and context match the checkpoint's. The registered
# decoder uses the shared batch processor and declares the existing
# positional input layout. The checkpoint loads through rules, which drop the
# tensors of the drafter and audio tokenizer the repo ships in subfolders.
mimo_v2_modulev3_arch = SupportedArchitecture(
    name="MiMoV2ForCausalLM",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["XiaomiMiMo/MiMo-V2.6-Flash-RL"],
    format_spec=Path(__file__).parent / "formatspec.json",
    default_weights_format=WeightsFormat.safetensors,
    # This package's config: it validates the dense-MLP scale-granule split
    # the V3 build needs at plan time, where the graph-API config would
    # accept a degree the build later rejects.
    default_encoding=MiMoV2Config.DEFAULT_ENCODING,
    supported_encodings=MiMoV2Config.SUPPORTED_ENCODINGS,
    pipeline_model=None,
    modules={GraphRole.DECODER: MiMoV2Decoder},
    tokenizer=TextTokenizer,
    context_type=TextContext,
    weight_rules=mimo_v2_weight_rules,
    use_weight_rules=True,
    config=MiMoV2Config,
    batching=None,
    multi_gpu_supported=True,
    supports_overlap_scheduler=False,
    supports_device_graph_capture=False,
    # MiMo-V2's DFlash drafter declares the Llama 3 draft's name, but it is
    # sharded like its target and stored in a format of its own.
    draft_architectures={
        "dflash": {"DFlashDraftModel": "DFlashMiMoV2DraftModel"}
    },
    # Mach finds the fused graph by this name.
    spec_decode_graph_names={"dflash": "unified_dflash_mimo_v2"},
    supports_spec_decode_mixed_batches=True,
)
