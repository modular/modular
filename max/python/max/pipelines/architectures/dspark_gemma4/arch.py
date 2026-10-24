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
"""Registers the native DSpark draft checkpoint's architecture."""

from __future__ import annotations

from max.graph.weights import WeightsFormat
from max.pipelines.context import TextContext
from max.pipelines.lib import SupportedArchitecture, TextTokenizer
from max.pipelines.modeling.types import PipelineTask

from .model_config import Gemma4DSparkDraftArchConfig, gemma4_dspark_12b_width

__all__ = ["gemma4_dspark_draft_arch"]

# The native DSpark draft checkpoint declares architectures
# ["Gemma4DSparkModel"]. It is only served as a --draft-model beside a
# Gemma 4 target, so it has no pipeline model: its config builds the draft for a unified
# spec-decode graph, supplies the draft's KV leaf, and the width it drafts at
# comes from its checkpoint.
gemma4_dspark_draft_arch = SupportedArchitecture(
    name="Gemma4DSparkModel",
    example_repo_ids=[
        "deepseek-ai/dspark_gemma4_12b_block7",
    ],
    default_encoding="bfloat16",
    supported_encodings={"bfloat16"},
    pipeline_model=None,
    context_type=TextContext,
    tokenizer=TextTokenizer,
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=False,
    task=PipelineTask.TEXT_GENERATION,
    config=Gemma4DSparkDraftArchConfig,
    checkpoint_draft_width=gemma4_dspark_12b_width,
)
