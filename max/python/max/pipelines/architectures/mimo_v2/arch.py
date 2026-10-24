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

from max.graph.weights import WeightsFormat
from max.pipelines.architectures.gpt_oss.batch_processor import (
    GptOssBatchProcessor,
)
from max.pipelines.context import TextContext
from max.pipelines.kv_cache.memory_planner import PagedMemoryPlanner
from max.pipelines.lib import SupportedArchitecture, TextTokenizer
from max.pipelines.modeling.types import PipelineTask

from .model import MiMoV2Model
from .model_config import MiMoV2Config
from .weight_adapters import convert_safetensor_state_dict

mimo_v2_arch = SupportedArchitecture(
    name="MiMoV2ForCausalLM",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["XiaomiMiMo/MiMo-V2.6-Flash-RL"],
    default_weights_format=WeightsFormat.safetensors,
    default_encoding=MiMoV2Config.DEFAULT_ENCODING,
    supported_encodings=MiMoV2Config.SUPPORTED_ENCODINGS,
    pipeline_model=MiMoV2Model,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    weight_adapters={
        WeightsFormat.safetensors: convert_safetensor_state_dict,
    },
    config=MiMoV2Config,
    batching=GptOssBatchProcessor,
    multi_gpu_supported=True,
    # The checkpoint loads as stored, so its files size the weights.
    memory_planner=PagedMemoryPlanner.with_activation_reservation(
        0, always_signal_buffers=True
    ),
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
