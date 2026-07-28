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

from __future__ import annotations

from max.driver import Buffer
from max.dtype import DType
from max.nn.kv_cache.cache_params import KVCacheMemory


def view_2d_uint8(buf: Buffer, total_num_pages: int) -> Buffer:
    """Views a raw buffer as a 2-D uint8 ``[total_num_pages, bytes_per_page]`` array."""
    bytes_per_page = (
        buf.num_elements * buf.dtype.size_in_bytes // total_num_pages
    )
    return buf.view(DType.uint8, [total_num_pages, bytes_per_page])


def kv_memory(buf: Buffer, total_num_pages: int) -> KVCacheMemory:
    """Wraps a raw buffer as a 2-D uint8 ``KVCacheMemory`` unit."""
    return KVCacheMemory(buffer=view_2d_uint8(buf, total_num_pages))
