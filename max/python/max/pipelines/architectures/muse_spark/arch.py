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
from max.pipelines.context import TextContext
from max.pipelines.lib import SupportedArchitecture, TextTokenizer
from max.pipelines.modeling.types import PipelineTask

from .model import MuseSparkModel
from .model_config import MuseSparkConfig

muse_spark_arch = SupportedArchitecture(
    name="MuseSparkForConditionalGeneration_ModuleV3",
    example_repo_ids=["meta-models/Muse-Spark-Open-816B-A42B-NVFP4"],
    default_encoding=MuseSparkConfig.DEFAULT_ENCODING,
    supported_encodings=MuseSparkConfig.SUPPORTED_ENCODINGS,
    pipeline_model=MuseSparkModel,
    task=PipelineTask.TEXT_GENERATION,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    default_weights_format=WeightsFormat.safetensors,
    config=MuseSparkConfig,
)
