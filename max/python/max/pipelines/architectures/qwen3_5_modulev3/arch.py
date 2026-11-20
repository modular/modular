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
# Ported to ModuleV3 from
# max/python/max/pipelines/architectures/qwen3_5/arch.py at main 4df8111afba.
"""Registers Qwen3.5 on ModuleV3.

Selected over the graph-API build by ``prefer_module_v3``; both read the same
checkpoints and the same cache tree.
"""

from __future__ import annotations

from pathlib import Path

from max.graph.weights import WeightsFormat
from max.pipelines.lib import GraphRole, SupportedArchitecture
from max.pipelines.modeling.types import InputModality, PipelineTask

from .batch_processor import Qwen3_5BatchProcessor
from .context import Qwen3_5TextAndVisionContext
from .model_config import Qwen3_5Config
from .qwen3_5 import Qwen3_5Decoder
from .reasoning import Qwen3_5ReasoningParser  # noqa: F401  registers "qwen3_5"
from .tokenizer import Qwen3_5Tokenizer
from .tool_parser import Qwen3_5ToolParser  # noqa: F401  registers "qwen3_5"
from .vision import Qwen3_5VisionEncoder
from .weight_adapters import qwen3_5_weight_rules

__all__ = ["qwen3_5_modulev3_arch", "qwen3_5_text_modulev3_arch"]


def _architecture(
    name: str,
    example_repo_ids: list[str],
    input_modalities: set[InputModality],
) -> SupportedArchitecture:
    """Returns a Qwen3.5 registration under ``name``.

    A text-only checkpoint builds no vision tower: the ``VISION`` module
    turns itself off for a config without one.
    """
    return SupportedArchitecture(
        name=name,
        task=PipelineTask.TEXT_GENERATION,
        example_repo_ids=example_repo_ids,
        default_weights_format=WeightsFormat.safetensors,
        default_encoding=Qwen3_5Config.DEFAULT_ENCODING,
        supported_encodings=Qwen3_5Config.SUPPORTED_ENCODINGS,
        modules={
            GraphRole.DECODER: Qwen3_5Decoder,
            GraphRole.VISION: Qwen3_5VisionEncoder,
        },
        tokenizer=Qwen3_5Tokenizer,
        context_type=Qwen3_5TextAndVisionContext,
        input_modalities=input_modalities,
        weight_rules=qwen3_5_weight_rules,
        use_weight_rules=True,
        # Declaring the state leaves is what selects the Jenga pool.
        checkpoints_recurrent_state=True,
        # The state checkpoints only at a page boundary, which a step's
        # accepted drafts move past.
        prefix_caching_under_spec=False,
        config=Qwen3_5Config,
        batching=Qwen3_5BatchProcessor,
        multi_gpu_supported=True,
        tool_parser="qwen3_5",
        format_spec=Path(__file__).parent / "formatspec.json",
        reasoning_parser="qwen3_5",
        # A warmup probe's state pages are freed with its KV, so a sweep of
        # probes cannot exhaust them.
        supports_device_graph_capture=True,
        checkpoint_drafts={"mtp": "MTPQwen3_5DraftModel"},
        # Mach binds the fused graphs by the graph-API pairings' names.
        spec_decode_graph_names={
            "dflash2": "qwen3_5_with_dflash2_graph",
            "mtp": "qwen3_5_with_mtp_graph",
        },
    )


qwen3_5_modulev3_arch = _architecture(
    "Qwen3_5ForConditionalGeneration_ModuleV3",
    ["Qwen/Qwen3.5-27B", "Qwen/Qwen3.8-27B"],
    {InputModality.TEXT, InputModality.IMAGE, InputModality.VIDEO},
)

qwen3_5_text_modulev3_arch = _architecture(
    "Qwen3_5ForCausalLM_ModuleV3",
    ["Mapika/decider-0.8b", "Mapika/decider-2b"],
    {InputModality.TEXT},
)
