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

import asyncio

import numpy as np
from max.driver import Accelerator
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.kv_cache import PagedKVCacheManager
from max.nn.legacy.kv_cache import KVCacheInputs, KVCacheParams
from test_common.context_utils import create_text_context


def test_kv_cache_gpu() -> None:
    asyncio.run(_test_kv_cache_gpu())


async def _test_kv_cache_gpu() -> None:
    device = Accelerator()
    kv_params = KVCacheParams(
        n_kv_heads=8,
        head_dim=128,
        dtype=DType.bfloat16,
        num_layers=32,
        cache_strategy="paged",
        page_size=128,
        devices=[DeviceRef.GPU()],
    )
    kv_manager = PagedKVCacheManager(
        params=kv_params,
        session=InferenceSession(devices=[device]),
        total_num_pages=8,
        max_batch_size=128,
    )
    context = create_text_context(np.empty(1))
    kv_manager.claim(context.request_id, replica_idx=0)
    kv_manager.alloc(context, replica_idx=0, num_steps=1)
    batch = [context]
    # suffixed [0] because we only have one device
    kv_tuple = kv_manager.runtime_inputs([batch])[0]
    assert isinstance(kv_tuple, KVCacheInputs)
    assert len(kv_tuple) == 5
