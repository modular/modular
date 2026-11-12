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
"""DeepSeek-V4 building blocks on ModuleV3."""

from .attention import DeepseekV4Attention
from .cache import KEY, VALUE, CacheLeaf, DeepseekV4Cache, LayerIndex
from .compressor import DeepseekV4Compressor
from .csa import CompressedStream, compressed_stream
from .hadamard import hadamard_rotate
from .hyper_connection import (
    PendingPost,
    Streams,
    expand_copies,
    hc_head,
    hc_merge,
    hc_post,
    hc_pre_norm,
    hyper_connection,
)
from .indexer import DeepseekV4Indexer
from .moe import (
    DeepseekV4Gate,
    DeepseekV4MoE,
    DeepseekV4RoutedExperts,
    DeepseekV4SharedExpert,
)
from .norm import DeepseekV4RMSNorm, weightless_rms_normalize
from .quantization import (
    DeepseekV4Fp8Linear,
    block_scale,
    dequantize_fp8_blocks,
    fp4_qat_quantize,
    fp8_qat_quantize,
    linear_for,
)
from .ragged import RaggedRows, WindowRows
from .rope import DeepseekV4RotaryEmbedding, apply_rope_tail, rope_for_layer

__all__ = [
    "KEY",
    "VALUE",
    "CacheLeaf",
    "CompressedStream",
    "DeepseekV4Attention",
    "DeepseekV4Cache",
    "DeepseekV4Compressor",
    "DeepseekV4Fp8Linear",
    "DeepseekV4Gate",
    "DeepseekV4Indexer",
    "DeepseekV4MoE",
    "DeepseekV4RMSNorm",
    "DeepseekV4RotaryEmbedding",
    "DeepseekV4RoutedExperts",
    "DeepseekV4SharedExpert",
    "LayerIndex",
    "PendingPost",
    "RaggedRows",
    "Streams",
    "WindowRows",
    "apply_rope_tail",
    "block_scale",
    "compressed_stream",
    "dequantize_fp8_blocks",
    "expand_copies",
    "fp4_qat_quantize",
    "fp8_qat_quantize",
    "hadamard_rotate",
    "hc_head",
    "hc_merge",
    "hc_post",
    "hc_pre_norm",
    "hyper_connection",
    "linear_for",
    "rope_for_layer",
    "weightless_rms_normalize",
]
