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
"""Drafter checkpoint -> MAX weights for the ModuleV3 MiMo-V2 DFlash drafter.

The drafter's tensors are all BF16, so the graph-API drafter's checkpoint
loading -- the shape and dtype checks, the trained ``mask_embedding.pt``
reader and the strict all-tensors-consumed rule -- loads the ModuleV3 module
unchanged; only the module that consumes the tensors differs. Re-exported
here so the drafter's Python surface is this package's, as for the base
architecture.
"""

from __future__ import annotations

from ..dflash_mimo_v2.weight_adapters import (  # re-export
    DRAFT_WEIGHTS_FILE,
    MASK_EMBEDDING,
    MASK_EMBEDDING_FILE,
    convert_safetensor_state_dict,
    drafter_tensor_shapes,
    load_mask_embedding,
)

__all__ = [
    "DRAFT_WEIGHTS_FILE",
    "MASK_EMBEDDING",
    "MASK_EMBEDDING_FILE",
    "convert_safetensor_state_dict",
    "drafter_tensor_shapes",
    "load_mask_embedding",
]
