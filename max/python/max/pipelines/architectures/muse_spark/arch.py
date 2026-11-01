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
"""Registers MuseSpark with the pipeline registry."""

from max.graph.weights import WeightsFormat
from max.pipelines.architectures.muse_glimmer.batch_processor import (
    MuseGlimmerBatchProcessor,
)
from max.pipelines.architectures.muse_glimmer.tokenizer import (
    MuseGlimmerTokenizer,
)
from max.pipelines.context import TextAndVisionContext
from max.pipelines.lib import SupportedArchitecture
from max.pipelines.modeling.types import PipelineTask

from . import weight_adapters
from .memory_planner import MuseSparkMemoryPlanner
from .model import MuseSparkModel
from .model_config import MuseSparkConfig

muse_spark_arch = SupportedArchitecture(
    name="MuseSparkForConditionalGeneration_ModuleV3",
    example_repo_ids=["meta-models/Muse-Spark-Open-816B-A42B-NVFP4"],
    default_encoding=MuseSparkConfig.DEFAULT_ENCODING,
    supported_encodings=MuseSparkConfig.SUPPORTED_ENCODINGS,
    pipeline_model=MuseSparkModel,
    task=PipelineTask.TEXT_GENERATION,
    tokenizer=MuseGlimmerTokenizer,
    context_type=TextAndVisionContext,
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=False,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_language_state_dict,
    },
    config=MuseSparkConfig,
    batching=MuseGlimmerBatchProcessor,
    memory_planner=MuseSparkMemoryPlanner,
    supports_overlap_scheduler=True,
    supports_device_graph_capture=False,
)
