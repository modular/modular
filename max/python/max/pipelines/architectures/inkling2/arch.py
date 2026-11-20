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
"""Inkling 2 architecture registration."""

from __future__ import annotations

from typing import Any

from max.graph.weights import WeightsFormat
from max.pipelines.context import TextAndVisionContext
from max.pipelines.lib import GraphRole, SupportedArchitecture
from max.pipelines.modeling.types import InputModality, PipelineTask

from .batch_processor import Inkling2BatchProcessor
from .dspark_config import Inkling2DSparkConfig
from .dspark_draft import (
    inkling2_dspark_draft_arch as make_inkling2_dspark_draft_arch,
)
from .model_config import Inkling2Config
from .modules import Inkling2Decoder
from .reasoning import (
    InklingReasoningParser,  # noqa: F401  registers "inkling2"
)
from .tokenizer import Inkling2Tokenizer
from .tool_parser import (
    InklingToolParser,  # noqa: F401  registers "inkling2"
)
from .vision import Inkling2Vision
from .weight_adapters import inkling2_weight_rules


def inkling2_dspark_draft_width(
    speculative: Any,
    target_huggingface_config: Any,
    draft_huggingface_config: Any,
) -> int:
    """Returns the DSpark draft width this checkpoint was trained for.

    The drafter ships in the checkpoint, so its block is the target config's
    ``mtp_config``: one draft token per mask row, ``block_size - 1``.

    Args:
        speculative: The pipeline's speculative configuration.
        target_huggingface_config: The target's Hugging Face config.
        draft_huggingface_config: The draft's, unread: this drafter has no
            config of its own.

    Returns:
        The width, which becomes ``num_speculative_tokens``.

    Raises:
        ValueError: If a different ``num_speculative_tokens`` is requested.
    """
    del draft_huggingface_config
    width = Inkling2DSparkConfig.from_hf(
        target_huggingface_config
    ).num_speculative_tokens
    requested = speculative.num_speculative_tokens
    if requested is not None and requested != width:
        raise ValueError(
            f"Inkling 2's DSpark block drafts exactly {width} tokens per step,"
            " one per mask row of its trained block; got"
            f" num_speculative_tokens={requested}."
        )
    return width


inkling2_arch = SupportedArchitecture(
    name="Inkling2ForConditionalGeneration_ModuleV3",
    task=PipelineTask.TEXT_GENERATION,
    input_modalities={InputModality.TEXT, InputModality.IMAGE},
    example_repo_ids=["thinkingmachines/Inkling-2-Small"],
    default_weights_format=WeightsFormat.safetensors,
    default_encoding=Inkling2Config.DEFAULT_ENCODING,
    supported_encodings=Inkling2Config.SUPPORTED_ENCODINGS,
    modules={
        GraphRole.DECODER: Inkling2Decoder,
        GraphRole.VISION: Inkling2Vision,
    },
    batching=Inkling2BatchProcessor,
    tokenizer=Inkling2Tokenizer,
    context_type=TextAndVisionContext,
    config=Inkling2Config,
    tool_parser="inkling2",
    reasoning_parser="inkling2",
    weight_rules=inkling2_weight_rules,
    use_weight_rules=True,
    multi_gpu_supported=True,
    checkpoints_recurrent_state=True,
    checkpoint_draft_width=inkling2_dspark_draft_width,
    checkpoint_drafts={"dspark": "Inkling2DSparkDraftModel_ModuleV3"},
    # The serving engine boots the fused graph by this name.
    spec_decode_graph_names={"dspark": "UnifiedDSparkInkling2ModuleV3"},
)

inkling2_dspark_draft_arch = make_inkling2_dspark_draft_arch(inkling2_arch)
