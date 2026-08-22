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

"""Host-tier block sizing for KV cache parameter trees."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from max.nn.kv_cache import NullKVConnectorConfig, host_bytes_per_block
from max.nn.kv_cache.cache_params import MultiKVCacheParams


def _leaf(bytes_per_block: int, replicated: bool, tp: int) -> Any:
    return SimpleNamespace(
        bytes_per_block=bytes_per_block,
        replicates_kv_across_tp=replicated,
        tensor_parallel_degree=tp,
        page_size=128,
        data_parallel_degree=1,
        devices=[],
        kv_connector_config=NullKVConnectorConfig(),
        n_devices=1,
    )


def test_leaf_divides_only_when_replicated() -> None:
    assert host_bytes_per_block(_leaf(4096, replicated=False, tp=4)) == 4096
    assert host_bytes_per_block(_leaf(4096, replicated=True, tp=4)) == 1024


class _Tree(MultiKVCacheParams):
    """A tree with only what ``host_bytes_per_block`` reads.

    Bypasses the dataclass __init__ deliberately: its validation checks child
    consistency this test does not exercise, and satisfying it would need a
    growing list of unrelated fields.
    """

    def __init__(self, children: dict[str, Any]) -> None:
        # ``children`` is read-only on the frozen dataclass.
        object.__setattr__(self, "children", children)


def test_tree_applies_each_child_s_own_replication() -> None:
    """An MLA target beside an MHA draft: only the target is replicated.

    ``MultiKVCacheParams`` reports its FIRST child's replication, so dividing
    the whole tree by that degree would undercount the sharded sibling.
    """
    tree = _Tree(
        {
            "target": _leaf(4096, replicated=True, tp=4),
            "draft": _leaf(512, replicated=False, tp=4),
        }
    )

    # target 4096/4 = 1024, draft 512 unchanged -> 1536, not 4608/4 = 1152.
    assert host_bytes_per_block(tree) == 1536
