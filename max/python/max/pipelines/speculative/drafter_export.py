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
"""What a speculative graph baked in from its drafter, for its export sidecar."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from max.pipelines.lib import PipelineConfig

__all__ = ["DrafterExport", "sampleable_vocab_size"]


@dataclass(frozen=True)
class DrafterExport:
    """What an export built into the graph from the drafter.

    An exporter records it in the graph's sidecar, with the hashes of the
    files in ``directory`` for the serving engine to check at load time.
    """

    directory: Path
    """The drafter directory whose files the graph was built from."""
    target_layer_ids: list[int]
    num_speculative_tokens: int
    """The verify width of the speculative graph."""
    spec_block_size: int
    """The drafter's block width."""


def sampleable_vocab_size(pipeline_config: PipelineConfig) -> int:
    """Returns the tokenizer's size; ``lm_head`` rows past it are padding."""
    # Imported here: the registry module imports this package, so a
    # module-level import would cycle.
    from max.pipelines.lib.registry import PIPELINE_REGISTRY

    return len(
        PIPELINE_REGISTRY.get_active_tokenizer(
            pipeline_config.model.huggingface_model_repo
        )
    )
