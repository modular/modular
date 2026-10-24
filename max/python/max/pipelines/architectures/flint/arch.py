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

"""Registers Flint with the pipeline registry."""

from max.graph.weights import WeightsFormat
from max.pipelines.context import TextContext
from max.pipelines.kv_cache.memory_planner import PagedMemoryPlanner
from max.pipelines.lib import SupportedArchitecture, TextTokenizer
from max.pipelines.modeling.types import PipelineTask

from . import weight_adapters
from .batch_processor import FlintBatchProcessor
from .model import FlintModel
from .model_config import FlintConfig

flint_arch = SupportedArchitecture(
    name="FlintForCausalLM_ModuleV3",
    example_repo_ids=["RecursiveSI/flint"],
    default_encoding=FlintConfig.DEFAULT_ENCODING,
    supported_encodings=FlintConfig.SUPPORTED_ENCODINGS,
    pipeline_model=FlintModel,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=False,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
    task=PipelineTask.TEXT_GENERATION,
    config=FlintConfig,
    batching=FlintBatchProcessor,
    memory_planner=PagedMemoryPlanner,
)
