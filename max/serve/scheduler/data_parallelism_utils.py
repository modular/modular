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

from max.interfaces import RequestID, TextGenerationInputs
from max.nn.kv_cache import MultiPagedKVCacheManager, PagedKVCacheManager
from max.pipelines.core import TextContext


def split_by_replica_idx(
    inputs: TextGenerationInputs[TextContext],
    num_replicas: int,
    paged_cache: PagedKVCacheManager | None = None,
) -> None:
    """Splits a batch into a list of batches."""
    if num_replicas == 1:
        inputs.batches = [inputs.batch]
        return

    assert isinstance(paged_cache, MultiPagedKVCacheManager)

    batches: list[dict[RequestID, TextContext]] = [
        {} for _ in range(num_replicas)
    ]

    # First pass: place requests that already have a replica idx
    for req_id, context in inputs.batch.items():
        replica_idx = paged_cache.get_replica(context)
        batches[replica_idx][req_id] = context
    inputs.batches = batches
