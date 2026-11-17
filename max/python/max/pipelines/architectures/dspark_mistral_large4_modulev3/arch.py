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

"""Supported architecture entry for the ModuleV3 Mistral Large 4 DSpark drafter."""

from max.graph.weights import WeightsFormat
from max.pipelines.architectures.mistral_large4_modulev3 import weight_adapters
from max.pipelines.architectures.mistral_large4_modulev3.arch import (
    ml4_dspark_draft_width,
)
from max.pipelines.architectures.mistral_large4_modulev3.config import (
    MistralLarge4Config,
)
from max.pipelines.context import TextContext
from max.pipelines.lib import SupportedArchitecture, TextTokenizer
from max.pipelines.modeling.types import PipelineTask

from .model import DSparkMistralLarge4DraftArchConfig

__all__ = ["dspark_mistral_large4_modulev3_draft_arch"]

# The drafter ships inside the target's checkpoint, which has no config of its
# own, so spec decoding names this architecture from the target's
# ``checkpoint_drafts``. It has no pipeline model: its weights arrive through
# the target's adapter and are renamed by its own config's
# ``checkpoint_weights``.
dspark_mistral_large4_modulev3_draft_arch = SupportedArchitecture(
    name="DSparkMistralLarge4DraftModel_ModuleV3",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["modular/Mistral-Large-4-1T-A52B-NVFP4-modulev3"],
    default_encoding=MistralLarge4Config.DEFAULT_ENCODING,
    supported_encodings=MistralLarge4Config.SUPPORTED_ENCODINGS,
    # Split over the target's devices: the latent cache is replicated and the
    # block runs on every rank.
    multi_gpu_supported=True,
    pipeline_model=None,
    tokenizer=TextTokenizer,
    # The drafter reads the target's residual streams, which are text only.
    context_type=TextContext,
    default_weights_format=WeightsFormat.safetensors,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
    config=DSparkMistralLarge4DraftArchConfig,
    # The fused pairing's architecture is the target's renamed, and it takes this
    # hook from the *draft* it paired with -- so the block width has to be
    # registered here as well, or the fused config never resolves the width the
    # target's leaf and this drafter's agree on.
    checkpoint_draft_width=ml4_dspark_draft_width,
    # Three dense layers and a recurrent head, all of them part of the fused
    # graph's capture.
    supports_device_graph_capture=True,
)
