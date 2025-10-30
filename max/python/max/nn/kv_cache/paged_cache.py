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

"""PagedCacheValues"""

from dataclasses import dataclass

from max.graph import (
    BufferValue,
    TensorValue,
)
from max.nn.kv_cache.nested_iterable import NestedIterableDataclass


@dataclass
class PagedCacheValues(NestedIterableDataclass):
    kv_blocks: BufferValue
    cache_lengths: TensorValue
    lookup_table: TensorValue
    max_lengths: TensorValue
