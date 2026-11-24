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
"""Registers Mistral Large 4 on ModuleV3 with the pipeline registry.

The checkpoint declares ``MistralLarge4ForConditionalGeneration``, and the
``_ModuleV3`` suffix is the registry's name for the eager ``Module`` build of
that architecture, as it is for GLM-5.3 and DeepSeek-V3.2.
"""

from pathlib import Path
from typing import Any

from max.graph.weights import WeightsFormat
from max.pipelines.context.context_validators import (
    validate_requires_vision_context,
)
from max.pipelines.lib import SupportedArchitecture
from max.pipelines.modeling.types import InputModality, PipelineTask

from ..deepseekV3_modulev3.batch_processor import (
    DeepseekV3ModuleV3BatchProcessor,
)
from . import weight_adapters
from .config import MistralLarge4Config
from .context import MistralLarge4VisionContext
from .model import MistralLarge4Model
from .model_config import (
    ML4_DSPARK_BLOCK_WIDTH,
    TOKENIZER_ARTIFACT_REVISION,
    TOKENIZER_REPO_ID,
)
from .tokenizer import MistralLarge4VisionTokenizer


def ml4_dspark_draft_width(
    speculative: Any,
    target_huggingface_config: Any,
    draft_huggingface_config: Any,
) -> int:
    """Returns the DSpark draft width this checkpoint was trained for.

    The ``mtp`` block that declares it (``mtp.depth``) ships in the checkpoint's
    ``params.json``, which this hook -- it sees the Hugging Face configs only --
    cannot open, so the width is pinned here and cross-checked where the
    drafter's own geometry is read: the draft architecture's ``build_module``
    compares ``--num-speculative-tokens`` with ``mtp.depth``, and
    :func:`~max.pipelines.architectures.mistral_large4_modulev3.dspark.
    dspark_speculative_slots` compares it with the leaf's block width.

    Args:
        speculative: The pipeline's speculative configuration.
        target_huggingface_config: The target's Hugging Face config, unread.
        draft_huggingface_config: The draft's, unread: this drafter ships in
            the target checkpoint and has no config of its own.

    Returns:
        The block width, which becomes ``num_speculative_tokens``.

    Raises:
        ValueError: If a different ``num_speculative_tokens`` is requested.
    """
    del target_huggingface_config, draft_huggingface_config
    requested = speculative.num_speculative_tokens
    if requested is not None and requested != ML4_DSPARK_BLOCK_WIDTH:
        raise ValueError(
            "Mistral Large 4's DSpark block drafts exactly"
            f" {ML4_DSPARK_BLOCK_WIDTH} tokens per step (the block's trained"
            " depth: every block row drafts one token, and the target verifies"
            f" one more); got num_speculative_tokens={requested}"
        )
    return ML4_DSPARK_BLOCK_WIDTH


def ml4_tokenizer_artifacts() -> tuple[str, str]:
    """The repo and revision this family's tokenizer files load from.

    The NVFP4 release ships weights and ``params.json``; its tokenizer files are
    the base release's, byte-identical, so both load one set at one pin and a
    consumer never reads whichever ``main`` a repo happens to be at.

    Returns:
        The base repo's id and the pinned revision of its tokenizer artifacts.
    """
    return TOKENIZER_REPO_ID, TOKENIZER_ARTIFACT_REVISION


mistral_large4_modulev3_arch = SupportedArchitecture(
    name="MistralLarge4ForConditionalGeneration_ModuleV3",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=[
        "mistralai/Mistral-Large-4-1T-A52B-NVFP4",
    ],
    # The checkpoint's own ModelOpt mixture: NVFP4 routed experts beside FP8
    # block-scaled attention, shared expert and layer 0, and bf16 MLA.
    default_encoding=MistralLarge4Config.DEFAULT_ENCODING,
    supported_encodings=MistralLarge4Config.SUPPORTED_ENCODINGS,
    # TP8 attention with EP8 routed experts on one 8x B200 node, the GLM-5.3
    # layout; the attention heads are split 8 per rank.
    multi_gpu_supported=True,
    pipeline_model=MistralLarge4Model,
    batching=DeepseekV3ModuleV3BatchProcessor,
    # Images ride the chat template's ``[IMG]`` placeholder, which the
    # tokenizer's processor grows into rows of image tokens closed by
    # break tokens and one end token -- the span the vision scatter fills.
    input_modalities={InputModality.TEXT, InputModality.IMAGE},
    tokenizer=MistralLarge4VisionTokenizer,
    # The NVFP4 release carries no tokenizer of its own that MAX should read:
    # its artifacts are the base repo's, pinned.
    tokenizer_artifacts=ml4_tokenizer_artifacts,
    context_type=MistralLarge4VisionContext,
    context_validators=[validate_requires_vision_context],
    default_weights_format=WeightsFormat.safetensors,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
    supports_empty_batches=True,
    requires_max_batch_context_length=True,
    config=MistralLarge4Config,
    format_spec=Path(__file__).parent / "formatspec.json",
    # The drafter ships inside this checkpoint as its ``mtp_dspark.*`` block, so
    # spec decoding names the draft architecture from the target's own.
    checkpoint_drafts={"dspark": "DSparkMistralLarge4DraftModel"},
    # The block's width is the checkpoint's, so the target's leaf and the
    # drafter's reserve the same number of draft slots per step.
    checkpoint_draft_width=ml4_dspark_draft_width,
    # Mach's block-spec executor binds the fused graph by this constant
    # (``block_spec_step_executor.rs``), so it cannot drift from MAX's.
    spec_decode_graph_names={"dspark": "UnifiedDSparkMistralLarge4Module"},
)
