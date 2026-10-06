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
"""Loading the DFlash drafter an export bakes in, and what it records of it."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from max.graph.weights import WeightData, load_weights
from max.pipelines.lib import HuggingFaceRepo

from ..dflash_mimo_v2 import (
    DFlashMiMoV2Config,
    convert_safetensor_state_dict,
    load_mask_embedding,
)
from ..dflash_mimo_v2.weight_adapters import (
    DRAFT_WEIGHTS_FILE,
    MASK_EMBEDDING,
    MASK_EMBEDDING_FILE,
)
from .model_config import repo_file


@dataclass(frozen=True)
class DrafterExport:
    """What an export built into the graph from the drafter.

    ``gen_mef`` records it in the sidecar with the hashes of the files in
    ``directory``, and Mach requires the spec and base-ctx records to agree.
    """

    directory: Path
    """The drafter directory whose files the graph was built from."""
    target_layer_ids: list[int]
    num_speculative_tokens: int | None = None
    """The verify width of a speculative graph; ``None`` for base-ctx."""
    spec_block_size: int | None = None
    """The drafter's block width in a speculative graph."""


def drafter_export(
    config: DFlashMiMoV2Config,
    directory: Path,
    *,
    num_speculative_tokens: int | None = None,
) -> DrafterExport:
    """Describes a graph built from ``config`` for its export sidecar."""
    return DrafterExport(
        directory=directory,
        target_layer_ids=list(config.target_layer_ids),
        num_speculative_tokens=num_speculative_tokens,
        spec_block_size=(
            None if num_speculative_tokens is None else config.block_size
        ),
    )


def load_drafter(
    repo: HuggingFaceRepo, config: DFlashMiMoV2Config, subfolder: str = ""
) -> tuple[dict[str, WeightData], WeightData, Path]:
    """Loads and checks every drafter tensor and the mask embedding.

    Returns:
        The drafter's tensors without the mask embedding, the mask embedding,
        and the directory they were read from.
    """
    prefix = f"{subfolder}/" if subfolder else ""
    weights_path = repo_file(repo, prefix + DRAFT_WEIGHTS_FILE)
    mask = load_mask_embedding(
        repo_file(repo, prefix + MASK_EMBEDDING_FILE), config
    )
    state = convert_safetensor_state_dict(
        dict(load_weights([weights_path]).items()), config, mask
    )
    del state[MASK_EMBEDDING]
    return state, mask, weights_path.parent
