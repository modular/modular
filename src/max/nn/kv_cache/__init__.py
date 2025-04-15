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

from .cache_params import KVCacheParams, KVCacheStrategy
from .continuous_batching_cache import (
    ContinuousBatchingKVCache,
    ContinuousBatchingKVCacheCollection,
    ContinuousBatchingKVCacheCollectionType,
    ContinuousBatchingKVCacheManager,
    ContinuousBatchingKVCacheType,
    FetchContinuousBatchingKVCacheCollection,
)
from .hf import ContinuousHFStaticCache
from .manager import (
    KVCacheInputs,
    KVCacheInputsSequence,
    KVCacheInputSymbols,
    KVCacheManager,
    PaddedKVCacheInputs,
    RaggedKVCacheInputs,
)
from .naive_cache import NaiveKVCacheManager
from .paged_cache import (
    BlockCopyOp,
    BlockCopyType,
    FetchPagedKVCacheCollection,
    FetchPagedKVCacheCollectionFA3Fallback,
    PagedKVCacheCollection,
    PagedKVCacheCollectionFA3Fallback,
    PagedKVCacheManager,
    PagedKVCacheManagerFA3Fallback,
    PagedKVCacheType,
)
from .registry import (
    estimate_kv_cache_size,
    infer_optimal_batch_size,
    load_kv_manager,
)
from .utils import build_max_lengths_tensor

__all__ = [
    "ContinuousBatchingKVCache",
    "ContinuousBatchingKVCacheCollection",
    "ContinuousBatchingKVCacheCollectionType",
    "ContinuousBatchingKVCacheManager",
    "ContinuousBatchingKVCacheType",
    "FetchContinuousBatchingKVCacheCollection",
    "NaiveKVCacheManager",
    "ContinuousHFStaticCache",
    "KVCacheStrategy",
    "KVCacheParams",
    "KVCacheInputs",
    "KVCacheInputsSequence",
    "KVCacheManager",
    "KVCacheInputSymbols",
    "PaddedKVCacheInputs",
    "RaggedKVCacheInputs",
    "BlockCopyOp",
    "BlockCopyType",
    "FetchPagedKVCacheCollection",
    "FetchPagedKVCacheCollectionFA3Fallback",
    "PagedKVCacheManager",
    "PagedKVCacheCollection",
    "PagedKVCacheCollectionFA3Fallback",
    "PagedKVCacheManagerFA3Fallback",
    "PagedKVCacheType",
    "load_kv_manager",
    "estimate_kv_cache_size",
    "infer_optimal_batch_size",
    "build_max_lengths_tensor",
]
