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
"""The attention mechanism used within the model."""

from .attention_with_rope import (
    AttentionWithRope,
    GGUFQAttentionWithRope,
    GPTQAttentionWithRope,
    TensorParallelAttentionWithRope,
)
from .interfaces import DistributedAttentionImpl
from .mask_config import (
    AttentionMaskVariant,
    MHAMaskConfig,
    MHAMaskVariant,
    PositionalEncodingVariant,
)
from .multi_latent_attention import (
    LatentAttentionWithRope,
    TensorParallelLatentAttentionWithRope,
)
from .multihead_attention import MultiheadAttention
from .ragged_attention import RaggedAttention

__all__ = [
    "AttentionMaskVariant",
    "AttentionWithRope",
    "DistributedAttentionImpl",
    "GGUFQAttentionWithRope",
    "GPTQAttentionWithRope",
    "LatentAttentionWithRope",
    "MHAMaskConfig",
    "MHAMaskVariant",
    "MultiheadAttention",
    "PositionalEncodingVariant",
    "RaggedAttention",
    "TensorParallelAttentionWithRope",
    "TensorParallelLatentAttentionWithRope",
]
