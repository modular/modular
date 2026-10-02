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
from max.pipelines.context import TextAndVisionContext
from max.pipelines.lib import SupportedArchitecture
from max.pipelines.modeling.types import InputModality, PipelineTask

from . import weight_adapters
from .batch_processor import MuseGlimmerBatchProcessor
from .memory_planner import MuseGlimmerMemoryPlanner
from .model import MuseGlimmerModel
from .model_config import MuseGlimmerConfig
from .reasoning import (
    MuseGlimmerReasoningParser,  # noqa: F401  registers "muse_glimmer"
)
from .tokenizer import MuseGlimmerTokenizer
from .tool_parser import (
    MuseGlimmerToolParser,  # noqa: F401  registers "muse_glimmer"
)

muse_glimmer_arch = SupportedArchitecture(
    name="MuseGlimmerForConditionalGeneration_ModuleV3",
    example_repo_ids=["meta-models/Muse-Glimmer-30B"],
    default_encoding=MuseGlimmerConfig.DEFAULT_ENCODING,
    supported_encodings=MuseGlimmerConfig.SUPPORTED_ENCODINGS,
    pipeline_model=MuseGlimmerModel,
    task=PipelineTask.TEXT_GENERATION,
    tokenizer=MuseGlimmerTokenizer,
    context_type=TextAndVisionContext,
    input_modalities={InputModality.TEXT, InputModality.IMAGE},
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=False,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_language_state_dict,
    },
    config=MuseGlimmerConfig,
    batching=MuseGlimmerBatchProcessor,
    memory_planner=MuseGlimmerMemoryPlanner,
    supports_overlap_scheduler=True,
    supports_device_graph_capture=True,
    tool_parser="muse_glimmer",
    reasoning_parser="muse_glimmer",
    default_structured_output_backend="xgrammar",
)
