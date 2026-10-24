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

from ..speculators_common.draft_config import (
    speculators_dspark_width,
)
from .model import DSparkDraftArchConfig

# Speculators-format DSpark drafters declare the generic
# architectures ["DSparkDraftModel"], shared across target model lines
# (Gemma4 31B today, e.g. GLM-5.2 dspark tomorrow). They are only served
# as a --draft-model: the config resolution rewrites the (target, DSpark
# draft) pair to the target's unified architecture, so this registration
# exists for draft-side name lookup, field validation, and the
# max-sequence-length clamp. The tokenizer and context_type are never used for
# a draft.
dspark_speculators_draft_arch = SupportedArchitecture(
    name="DSparkDraftModel",
    example_repo_ids=[
        "RedHatAI/gemma-4-31B-it-speculator.dspark",
    ],
    default_encoding="bfloat16",
    supported_encodings={"bfloat16"},
    pipeline_model=None,
    context_type=TextContext,
    tokenizer=TextTokenizer,
    default_weights_format=WeightsFormat.safetensors,
    # The targets this draft runs beside here call it on one device, and a
    # unified pairing runs on several only where its draft does. A target
    # that shards it renames it to a draft of its own.
    multi_gpu_supported=False,
    task=PipelineTask.TEXT_GENERATION,
    config=DSparkDraftArchConfig,
    checkpoint_draft_width=speculators_dspark_width,
)
