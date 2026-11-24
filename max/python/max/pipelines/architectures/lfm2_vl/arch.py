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
"""Lfm2VlForConditionalGeneration — SupportedArchitecture registration.

The name MUST be the exact HF `config.json` `architectures[0]` tag: that
is what the arch lookup matches when `max serve` loads a model.

The arch registers its two modules and no `pipeline_model`: the
framework's registered pipeline model constructs them from the
finalized config, compiles the two graphs, and runs each step's new
images through its vision cache into the decoder's `vision` merge
group. The decoder's conv windows ride the KV cache tree as a
recurrent-state leaf (`checkpoints_recurrent_state`), and the batch
processor packs the tower's per-tile tables for the images the cache
selects.
"""

from max.graph.weights import WeightsFormat
from max.pipelines.context import TextAndVisionContext
from max.pipelines.lib import GraphRole, SupportedArchitecture
from max.pipelines.modeling.types import InputModality, PipelineTask

from .batch_processor import Lfm2VlBatchProcessor
from .model_config import Lfm2VlConfig
from .network import Lfm2VlDecoder
from .tokenizer import Lfm2VlTokenizer
from .vision.tower import Lfm2VlVision
from .weight_adapters import lfm2vl_weight_rules

lfm2_vl_arch = SupportedArchitecture(
    name="Lfm2VlForConditionalGeneration",
    example_repo_ids=[
        "LiquidAI/d1-3B",
    ],
    default_encoding=Lfm2VlConfig.DEFAULT_ENCODING,
    supported_encodings=Lfm2VlConfig.SUPPORTED_ENCODINGS,
    modules={
        GraphRole.DECODER: Lfm2VlDecoder,
        GraphRole.VISION: Lfm2VlVision,
    },
    tokenizer=Lfm2VlTokenizer,
    context_type=TextAndVisionContext,
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=False,
    weight_rules=lfm2vl_weight_rules,
    use_weight_rules=True,
    task=PipelineTask.TEXT_GENERATION,
    input_modalities={InputModality.TEXT, InputModality.IMAGE},
    # Prefix caching would have to evict cached image rows with the
    # prefix; it stays off until that interaction is measured. Chunked
    # prefill stays at the engine default (on): the paged MHA kernel's
    # prefill scores workspace grows with heads x prompt^2, and this
    # family's 32-head text stack at 32k context would need 128 GiB in
    # one allocation, more than any single device's memory pool.
    required_arguments={
        "enable_prefix_caching": False,
    },
    config=Lfm2VlConfig,
    batching=Lfm2VlBatchProcessor,
    # The decoder's conv layers carry their window across steps in an
    # engine-managed pool (RecurrentStateParams leaf in the cache tree).
    checkpoints_recurrent_state=True,
    supports_overlap_scheduler=False,
    supports_device_graph_capture=False,
)

SUPPORTED_ARCHITECTURE = lfm2_vl_arch
