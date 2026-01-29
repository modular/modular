# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
"""Neural network modules for MAX.

This module provides the primary neural network building blocks for MAX
using eager tensor execution.

New API:
    - :class:`Module`: Base class for neural network modules
    - :class:`Linear`: Linear transformation layer
    - :class:`Embedding`: Vector embedding layer
    - :class:`Sequential`: Sequential container for modules
    - :func:`module_dataclass`: Decorator for creating module dataclasses

For legacy layer-based API, use ``max.nn.legacy``:
    >>> from max.nn.legacy import Module, Layer, Linear
    >>> from max.nn.legacy.attention import AttentionWithRope

Example:
    >>> from max.nn import Module, Linear, Embedding, Sequential, module_dataclass
    >>> from max.tensor import Tensor
"""

# New Module-based API (primary)
# Legacy submodule is available for backward compatibility
from . import legacy
from .embedding import Embedding
from .linear import Linear
from .module import Module, module_dataclass
from .norm import GemmaRMSNorm, LayerNorm, RMSNorm
from .rope import RotaryEmbedding, TransposedRotaryEmbedding
from .sequential import ModuleList, Sequential

__all__ = [
    "Embedding",
    "GemmaRMSNorm",
    "LayerNorm",
    "Linear",
    "Module",
    "ModuleList",
    "RMSNorm",
    "RotaryEmbedding",
    "Sequential",
    "TransposedRotaryEmbedding",
    "legacy",
    "module_dataclass",
]
