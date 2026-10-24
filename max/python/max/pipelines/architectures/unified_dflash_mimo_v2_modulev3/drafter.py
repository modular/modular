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
"""Loading the DFlash drafter an export bakes in, and what it records of it.

``gen_mef`` reads the :class:`DrafterExport` the model holds and records it
in the sidecar with the hashes of the files in ``directory``, which Mach
re-hashes at boot; the class is the graph-API export's, so one sidecar
reader serves either build.
"""

from __future__ import annotations

from pathlib import Path

from max.graph.weights import WeightData, load_weights
from max.pipelines.lib import HuggingFaceRepo

from ..dflash_mimo_v2 import DFlashMiMoV2Config
from ..dflash_mimo_v2.drafter import DrafterExport, drafter_export
from ..dflash_mimo_v2_modulev3.weight_adapters import (
    DRAFT_WEIGHTS_FILE,
    MASK_EMBEDDING,
    MASK_EMBEDDING_FILE,
    convert_safetensor_state_dict,
    load_mask_embedding,
)
from ..unified_dflash_mimo_v2_modulev3.model_config import repo_file

__all__ = ["DrafterExport", "drafter_export", "load_drafter"]


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
