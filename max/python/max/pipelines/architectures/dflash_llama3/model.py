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
"""Placeholder pipeline model for the DFlash standalone draft architecture.

The DFlash draft model (registered HuggingFace architecture name
``DFlashDraftModel``) is never invoked as a standalone pipeline — it is
always loaded and executed through :class:`UnifiedDflashLlama3Model` from
``unified_dflash_llama3``. This placeholder exists solely so MAX's
architecture registry can resolve the draft's ``architectures[0]``
during ``PipelineConfig`` validation when a DFlash recipe is used.
"""

from __future__ import annotations

from max.pipelines.lib import (
    ModelInputs,
    ModelOutputs,
)

from ..llama3.model import LlamaModelBase


class DFlashLlama3Model(LlamaModelBase):
    """Placeholder pipeline model for the DFlash draft architecture.

    See module docstring. ``execute`` raises because the draft is only
    ever run via the unified pipeline.
    """

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        raise NotImplementedError(
            "DFlashLlama3Model is a placeholder for architecture-registry"
            " lookup. The DFlash draft is run through"
            " UnifiedDflashLlama3Model in the unified pipeline."
        )
