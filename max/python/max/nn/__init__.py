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
"""Neural network modules for MAX.

Graph-based API:
    >>> from max.nn import Linear, AttentionWithRope
    >>> from max.nn.kv_cache import KVCacheParams

Eager tensor API (module_v3):
    >>> from max.nn.module_v3 import Module, Linear, Embedding
"""

from . import module_v3
from .attention import (
    AttentionWithRope,
    DistributedAttentionImpl,
    GGUFQAttentionWithRope,
    GPTQAttentionWithRope,
    LatentAttentionWithRope,
    MultiheadAttention,
    RaggedAttention,
    TensorParallelAttentionWithRope,
    TensorParallelLatentAttentionWithRope,
)
from .clamp import clamp
from .comm import Allreduce, Signals
from .conv import Conv1D, Conv2d, Conv3D
from .conv_transpose import ConvTranspose1d, WeightNormConvTranspose1d
from .data_parallelism import split_batch, split_batch_replicated
from .embedding import Embedding, VocabParallelEmbedding
from .float8_config import (
    Float8Config,
    Float8InputScaleSpec,
    Float8ScaleGranularity,
    Float8ScaleOrigin,
    Float8WeightScaleSpec,
)
from .identity import Identity
from .kv_cache import (
    KVCacheInputs,
    KVCacheInputsSequence,
    KVCacheMetrics,
    KVCacheParams,
    KVCacheStrategy,
    PagedCacheValues,
    RaggedKVCacheInputs,
    build_max_lengths_tensor,
)
from .layer import Layer, LayerList, Module, Shardable
from .linear import (
    MLP,
    ColumnParallelLinear,
    GPTQLinear,
    Linear,
)
from .lora import AttentionWithRopeAndLoRA, LinearLoRA, SupportsLoRA
from .moe import MoE, MoEGate, MoEQuantized
from .norm import (
    ConstantLayerNorm,
    GroupNorm,
    LayerNorm,
    RMSNorm,
)
from .rotary_embedding import (
    DynamicRotaryEmbedding,
    LinearScalingParams,
    Llama3RopeScalingParams,
    Llama3RotaryEmbedding,
    LongRoPERotaryEmbedding,
    LongRoPEScalingParams,
    RotaryEmbedding,
    YarnRotaryEmbedding,
    YarnScalingParams,
)
from .sampling import (
    MinPSampler,
    RejectionSampler,
    RejectionSamplerWithResiduals,
)
from .sequential import Sequential
from .transformer import (
    DistributedTransformer,
    DistributedTransformerBlock,
    ReturnHiddenStates,
    ReturnLogits,
    Transformer,
    TransformerBlock,
)

__all__ = [
    "MLP",
    "Allreduce",
    "AttentionWithRope",
    "AttentionWithRopeAndLoRA",
    "ColumnParallelLinear",
    "ConstantLayerNorm",
    "Conv1D",
    "Conv2d",
    "Conv3D",
    "ConvTranspose1d",
    "DistributedAttentionImpl",
    "DistributedTransformer",
    "DistributedTransformerBlock",
    "DynamicRotaryEmbedding",
    "Embedding",
    "Float8Config",
    "Float8InputScaleSpec",
    "Float8ScaleGranularity",
    "Float8ScaleOrigin",
    "Float8WeightScaleSpec",
    "GGUFQAttentionWithRope",
    "GPTQAttentionWithRope",
    "GPTQLinear",
    "GroupNorm",
    "Identity",
    "KVCacheInputs",
    "KVCacheInputsSequence",
    "KVCacheMetrics",
    "KVCacheParams",
    "KVCacheStrategy",
    "LatentAttentionWithRope",
    "Layer",
    "LayerList",
    "LayerNorm",
    "Linear",
    "LinearLoRA",
    "LinearScalingParams",
    "Llama3RopeScalingParams",
    "Llama3RotaryEmbedding",
    "LongRoPERotaryEmbedding",
    "LongRoPEScalingParams",
    "MinPSampler",
    "MoE",
    "MoEGate",
    "MoEQuantized",
    "Module",
    "MultiheadAttention",
    "PagedCacheValues",
    "RMSNorm",
    "RaggedAttention",
    "RaggedKVCacheInputs",
    "RejectionSampler",
    "RejectionSamplerWithResiduals",
    "ReturnHiddenStates",
    "ReturnLogits",
    "RotaryEmbedding",
    "Sequential",
    "Shardable",
    "Signals",
    "SupportsLoRA",
    "TensorParallelAttentionWithRope",
    "TensorParallelLatentAttentionWithRope",
    "Transformer",
    "TransformerBlock",
    "VocabParallelEmbedding",
    "WeightNormConvTranspose1d",
    "YarnRotaryEmbedding",
    "YarnScalingParams",
    "build_max_lengths_tensor",
    "clamp",
    "module_v3",
    "split_batch",
    "split_batch_replicated",
]
