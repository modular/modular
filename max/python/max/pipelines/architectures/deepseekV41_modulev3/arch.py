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
"""Registers text generation through the DeepSeek-V4.1 ModuleV3 trunk."""

from max.graph.weights import WeightsFormat
from max.pipelines.context import TextContext
from max.pipelines.lib import SupportedArchitecture
from max.pipelines.modeling.types import PipelineTask

# Registers the ``deepseekv4`` reasoning parser at import.
from .. import deepseekV4 as _deepseekV4  # noqa: F401

# Registers the ``deepseekv41`` tool parser at import.
from ..deepseekV41 import tool_parser as _v41_tool_parser  # noqa: F401
from . import weight_adapters
from .batch_processor import DeepseekV41BatchProcessor
from .memory_planner import DeepseekV41MemoryPlanner
from .model import DeepseekV41Model
from .model_config import DeepseekV41Config
from .tokenizer import DeepseekV41Tokenizer

deepseekV41_modulev3_arch = SupportedArchitecture(
    name="DeepseekV41ForCausalLM_ModuleV3",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=[
        "nvidia/DeepSeek-V4.1-Flash-NVFP4",
    ],
    default_encoding=DeepseekV41Config.DEFAULT_ENCODING,
    supported_encodings=DeepseekV41Config.SUPPORTED_ENCODINGS,
    # Multi-GPU serves the data-parallel attention + expert-parallel MoE
    # shapes this trunk wires (dp == ep_size == the device count, the
    # full attention heads once per rank); plain tensor parallelism
    # still hits the window-list masking placement conflict.
    multi_gpu_supported=True,
    supports_device_graph_capture=False,
    supports_empty_batches=False,
    requires_max_batch_context_length=True,
    pipeline_model=DeepseekV41Model,
    tokenizer=DeepseekV41Tokenizer,
    context_type=TextContext,
    default_weights_format=WeightsFormat.safetensors,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
    # The trunk serves every request with a zero window start and a zero
    # write floor; a request resumed from cached pages would read its
    # window from position 0 and land compressor and indexer writes in
    # blocks the cached prefix shares with other sessions. Prefix caching
    # stays off until resuming from cached pages is supported.
    required_arguments={"enable_prefix_caching": False},
    config=DeepseekV41Config,
    batching=DeepseekV41BatchProcessor,
    memory_planner=DeepseekV41MemoryPlanner,
    reasoning_parser="deepseekv4",
    tool_parser="deepseekv41",
)
