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
"""A data-parallel degree above one must be refused, not half-honoured.

The caches are built per batch shard -- both
:class:`~max.nn.kv_cache.MLAKVCacheParams` leaves and the recurrent state take
``data_parallel_degree`` straight from the pipeline config -- while the compute
graph is tensor-parallel throughout: KDA and sparse MLA shard by head, and
``data_parallel_splits`` is declared as a graph input and then discarded. So a
batch shard reaches the cache and never reaches the graph, which reads and
writes as though it owned the whole batch.

Nothing downstream notices. ``--data-parallel-degree 8`` is the shape GLM-5.x
runs in production, so the failure mode this guards against is a crash at best
and wrong logits at worst, on the flag a deployment is most likely to set.
"""

from __future__ import annotations

import pytest
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import MLAKVCacheParams, MultiKVCacheParams
from max.pipelines.architectures.glm5_next.model_config import Glm5NextConfig

HIDDEN_SIZE = 256
INTERMEDIATE_SIZE = 512
NUM_HEADS = 8
PAGE_SIZE = 128
MAX_SEQ_LEN = 512


def _config(devices: list[DeviceRef], data_parallel_degree: int) -> None:
    leaf = MLAKVCacheParams(
        dtype=DType.bfloat16,
        head_dim=HIDDEN_SIZE,
        num_layers=1,
        page_size=PAGE_SIZE,
        devices=devices,
        num_q_heads=NUM_HEADS,
    )
    Glm5NextConfig(
        dtype=DType.bfloat16,
        kv_params=MultiKVCacheParams.from_params(
            {"mla": leaf, "indexer": leaf}
        ),
        devices=devices,
        max_seq_len=MAX_SEQ_LEN,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_HEADS,
        num_hidden_layers=1,
        qk_rope_head_dim=0,
        data_parallel_degree=data_parallel_degree,
    )


def test_tensor_parallel_degree_is_accepted() -> None:
    """The shape the architecture actually builds stays constructible."""
    _config([DeviceRef.GPU(0), DeviceRef.GPU(1)], data_parallel_degree=1)


@pytest.mark.parametrize("degree", [2, 8])
def test_data_parallel_degree_is_refused(degree: int) -> None:
    """The production flag value is the one that must not construct."""
    devices = [DeviceRef.GPU(i) for i in range(degree)]
    with pytest.raises(ValueError, match="tensor-parallel only"):
        _config(devices, data_parallel_degree=degree)
