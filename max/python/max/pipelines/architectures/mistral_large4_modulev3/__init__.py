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
"""Mistral Large 4 (ModuleV3).

``model_config`` turns the checkpoint's ``params.json`` into an explicit
per-layer config and emits the HF-style ``config.json``; ``weight_adapters``
owns the checkpoint-name-to-module-name table MAX and Mach must agree on;
``config`` and ``mistral_large4`` build the model on the DeepSeek-V3.2 sparse
MLA spine.
"""

from __future__ import annotations

from .arch import mistral_large4_modulev3_arch
from .config import (
    INDEXER_GROUP,
    NOPE_MLA_GROUP,
    ROPE_MLA_GROUP,
    MistralLarge4Config,
    build_kv_params,
    indexer_slot_map,
    kv_slot_map,
    mla_group,
    mla_head_dim,
)
from .mistral_large4 import MistralLarge4
from .model import MistralLarge4Model

__all__ = [
    "INDEXER_GROUP",
    "NOPE_MLA_GROUP",
    "ROPE_MLA_GROUP",
    "MistralLarge4",
    "MistralLarge4Config",
    "MistralLarge4Model",
    "build_kv_params",
    "indexer_slot_map",
    "kv_slot_map",
    "mistral_large4_modulev3_arch",
    "mla_group",
    "mla_head_dim",
]
