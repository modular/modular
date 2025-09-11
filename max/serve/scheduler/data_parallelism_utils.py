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
from __future__ import annotations

from typing import TypeVar

from max.interfaces.request import RequestID
from max.nn.kv_cache import (
    KVCacheAwareContext,
    MultiPagedKVCacheManager,
    PagedKVCacheManager,
)

T = TypeVar("T", bound=KVCacheAwareContext)
BatchType = dict[RequestID, T]


def split_by_replica_idx(
    batch: BatchType[T],
    num_replicas: int,
    paged_cache: PagedKVCacheManager[T] | None = None,
) -> list[BatchType[T]]:
    """Splits a batch into a list of batches."""
    if num_replicas == 1:
        return [batch]

    assert isinstance(paged_cache, MultiPagedKVCacheManager)

    batches: list[BatchType[T]] = [{} for _ in range(num_replicas)]

    # First pass: place requests that already have a replica idx
    for req_id, context in batch.items():
        replica_idx = paged_cache.get_replica(context)
        batches[replica_idx][req_id] = context
    return batches
