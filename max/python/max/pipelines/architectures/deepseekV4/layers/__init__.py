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
"""Layers specific to the DeepSeek-V4-Flash architecture."""

from .attention import DeepseekV4Attention
from .cache import CacheLeaf, DeepseekV4Cache
from .compressor import DeepseekV4Compressor
from .csa import CompressedStream, compressed_stream
from .dspark import (
    DSparkAttention,
    DSparkConfidenceHead,
    DSparkMarkovHead,
)
from .hadamard import hadamard_rotate
from .hyper_connection import (
    expand_copies,
    hc_head,
    hc_mix_width,
    hc_post,
    hc_pre,
    hc_split_sinkhorn,
)
from .indexer import DeepseekV4Indexer
from .moe import (
    DeepseekV4Expert,
    DeepseekV4Gate,
    DeepseekV4MoE,
    sqrt_softplus,
)
from .quantization import fp4_qat_quantize, fp8_qat_quantize
from .ragged import RaggedRows, WindowRows
from .rope import DeepseekV4RotaryEmbedding, apply_rope_tail, rope_for_layer
from .sparse_attention import sparse_attention

__all__ = [
    "CacheLeaf",
    "CompressedStream",
    "DSparkAttention",
    "DSparkConfidenceHead",
    "DSparkMarkovHead",
    "DeepseekV4Attention",
    "DeepseekV4Cache",
    "DeepseekV4Compressor",
    "DeepseekV4Expert",
    "DeepseekV4Gate",
    "DeepseekV4Indexer",
    "DeepseekV4MoE",
    "DeepseekV4RotaryEmbedding",
    "RaggedRows",
    "WindowRows",
    "apply_rope_tail",
    "compressed_stream",
    "expand_copies",
    "fp4_qat_quantize",
    "fp8_qat_quantize",
    "hadamard_rotate",
    "hc_head",
    "hc_mix_width",
    "hc_post",
    "hc_pre",
    "hc_split_sinkhorn",
    "rope_for_layer",
    "sparse_attention",
    "sqrt_softplus",
]
