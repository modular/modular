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
"""LFM3-Audio architecture registration."""

from __future__ import annotations

from max.graph.weights import WeightsFormat
from max.pipelines.context import TextAndVisionContext
from max.pipelines.kv_cache.memory_planner import PagedMemoryPlanner
from max.pipelines.lib import SupportedArchitecture
from max.pipelines.modeling.types import InputModality, PipelineTask

from ..lfm3_vl.tokenizer import Lfm3VlTokenizer
from .model import Lfm3AudioModel
from .model_config import Lfm3AudioConfig

lfm3_audio_arch = SupportedArchitecture(
    name="Lfm3AudioForConditionalGeneration",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["LiquidAI/LFM3-Audio-2B"],
    default_weights_format=WeightsFormat.safetensors,
    default_encoding=Lfm3AudioConfig.DEFAULT_ENCODING,
    supported_encodings=Lfm3AudioConfig.SUPPORTED_ENCODINGS,
    pipeline_model=Lfm3AudioModel,
    tokenizer=Lfm3VlTokenizer,
    context_type=TextAndVisionContext,
    input_modalities={InputModality.TEXT},
    config=Lfm3AudioConfig,
    batching=Lfm3AudioModel.batch_processor_cls,
    memory_planner=PagedMemoryPlanner.with_activation_reservation(4 * 1024**3),
    checkpoints_recurrent_state=True,
)
