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
"""Kimi K3 (ModuleV3 port).

Registered as ``KimiK3ForConditionalGeneration_ModuleV3``, selected over the
Graph-API architecture (``kimi_k3.kimi_k3_arch``) by ``prefer_module_v3``.

The KDA state is addressed by row id, as in ``kimi_k3`` (V2): each KDA layer
takes a conv and a recurrent pool (one per layer, or the same pooled leaf
passed to every layer) and its own ``[batch]`` row ids into each, the row of
that layer in the ``[num_layers, batch]`` tables ``state_cache.py`` declares.
"""

from __future__ import annotations

from .arch import kimi_k3_modulev3_arch
from .model import KimiK3Model
from .model_config import KimiK3Config

__all__ = [
    "KimiK3Config",
    "KimiK3Model",
    "kimi_k3_modulev3_arch",
]
