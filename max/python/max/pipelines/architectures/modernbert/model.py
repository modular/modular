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
"""Defines the ModernBERT pipeline model stub for architecture registration."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from max.driver import Buffer
from max.nn.kv_cache import KVCacheInputsInterface
from max.pipelines.context import TextContext
from max.pipelines.lib import ModelInputs, ModelOutputs, PipelineModel

from .model_config import ModernBertConfig


@dataclass
class ModernBertInputs(ModelInputs):
    """Input tensors for the ModernBERT model."""

    next_tokens_batch: Buffer
    attention_mask: Buffer


class ModernBertPipelineModel(PipelineModel[TextContext]):
    """Pipeline model stub; graph execution is implemented in a follow-up change."""

    model_config_cls: ClassVar[type[ModernBertConfig]] = ModernBertConfig

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        raise NotImplementedError(
            "ModernBERT graph execution is not implemented yet."
        )

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> ModernBertInputs:
        raise NotImplementedError(
            "ModernBERT input preparation is not implemented yet."
        )
