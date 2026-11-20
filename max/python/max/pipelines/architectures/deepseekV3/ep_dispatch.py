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
"""Expert-parallel dispatch decisions shared by the DeepseekV3 family."""

from __future__ import annotations

from max.dtype import DType
from max.nn.ep_config import calculate_ep_max_tokens_per_rank
from max.pipelines.lib.config import PipelineConfig
from max.pipelines.lib.config.model_config import _select_quantization_encoding
from max.pipelines.modeling.config_enums import (
    is_float4_encoding,
    supported_encoding_dtype,
)
from transformers import AutoConfig

from .model_config import DeepseekV3Config


def mtp_draft_ep_dispatch_dtype(
    pipeline_config: PipelineConfig,
) -> DType | None:
    """Returns the EP dispatch dtype the MTP draft uses, for an FP4 target.

    With an FP4 target the draft dispatches wider than the target does, so
    the shared EP buffers must be allocated for the draft. Returns ``None``
    when the override does not apply.
    """
    spec_config = pipeline_config.speculative
    if spec_config is None or not spec_config.is_mtp():
        return None

    encoding = _select_quantization_encoding(
        pipeline_config.model, DeepseekV3Config.DEFAULT_ENCODING
    )
    if not is_float4_encoding(encoding):
        return None

    if pipeline_config.draft_model is None:
        # MTP baked into the target checkpoint has no separate draft to read an
        # encoding from. FP4 checkpoints leave the NextN layer's routed experts
        # unquantized, so the draft dispatches through EP in bfloat16 and the
        # model upsizes the shared buffers to match (see
        # ``UnifiedMTPGlm5_2Model``).
        return DType.bfloat16

    draft_encoding = _select_quantization_encoding(
        pipeline_config.draft_model, DeepseekV3Config.DEFAULT_ENCODING
    )
    return supported_encoding_dtype(draft_encoding)


def ep_max_rank_send_tokens_for_pipeline(
    pipeline_config: PipelineConfig,
) -> int:
    """Upper bound on EP dispatch tokens held on one rank for this pipeline."""
    return calculate_ep_max_tokens_per_rank(
        max_batch_input_tokens=pipeline_config.runtime.max_batch_input_tokens,
        ep_size=pipeline_config.runtime.ep_size,
        data_parallel_degree=pipeline_config.model.data_parallel_degree,
        use_allreduce=pipeline_config.runtime.ep_use_allreduce,
    )


def ep_fuse_ffn_combine_send_for_pipeline(
    pipeline_config: PipelineConfig,
    huggingface_config: AutoConfig,
) -> bool:
    """Whether the MoE FFN should send its output straight to the peers.

    Conditions only the layer can see raise there instead of narrowing here.

    Defaults on except for a decode-only worker. Allocation is a graph-level
    property, so a role serving both phases gets one answer for both.

    Args:
        pipeline_config: Pipeline configuration.
        huggingface_config: HuggingFace model configuration.

    Returns:
        Whether to fuse the combine send into the FFN epilogue.
    """
    explicit = pipeline_config.runtime.ep_fuse_ffn_combine_send
    requested = (
        explicit
        if explicit is not None
        else pipeline_config.runtime.pipeline_role != "decode_only"
    )
    if not requested:
        return False

    # Allreduce routes within the device and never writes peer buffers.
    if pipeline_config.runtime.ep_use_allreduce:
        return False

    # The send has already done the async half of the combine, so it needs the
    # split async/wait pair. The EP forward only takes that path when an
    # unfused shared expert gives it something to overlap.
    if getattr(huggingface_config, "n_shared_experts", 0) != 1:
        return False

    return is_float4_encoding(
        _select_quantization_encoding(
            pipeline_config.model, DeepseekV3Config.DEFAULT_ENCODING
        )
    )
