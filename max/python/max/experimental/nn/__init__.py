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
"""Module implementation using eager tensors."""

from .activations import (
    GEGLU,
    GELU,
    ApproximateGELU,
    FeedForward,
    Identity,
    SiLU,
    SwiGLU,
)
from .conv import Conv1d, Conv2d, Conv3d, ConvTranspose1d
from .dropout import Dropout
from .embedding import Embedding
from .linear import Linear
from .module import Module, module_dataclass
from .norm import GemmaRMSNorm, GroupNorm, LayerNorm, RMSNorm
from .rope import RotaryEmbedding, TransposedRotaryEmbedding
from .sequential import ModuleList, Sequential

__all__ = [
    "GEGLU",
    "GELU",
    "ApproximateGELU",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "Dropout",
    "Embedding",
    "FeedForward",
    "GemmaRMSNorm",
    "GroupNorm",
    "Identity",
    "LayerNorm",
    "Linear",
    "Module",
    "ModuleList",
    "RMSNorm",
    "RotaryEmbedding",
    "Sequential",
    "SiLU",
    "SwiGLU",
    "TransposedRotaryEmbedding",
    "module_dataclass",
]
