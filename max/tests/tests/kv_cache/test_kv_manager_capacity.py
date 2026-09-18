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

"""Per-manager request capacity: uniform pages versus a Jenga slab."""

from __future__ import annotations

from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import (
    MHAKVCacheParams,
    MultiKVCacheParams,
    RecurrentStateParams,
    RecurrentStateRegion,
    compute_max_seq_len_fitting_in_cache,
)
from max.pipelines.kv_cache import (
    JengaKVCacheManager,
    max_seq_len_fitting_in_cache,
)
from max.pipelines.kv_cache.paged_kv_cache.jenga_block_pool import (
    JengaBlockPool,
)
from max.support.math import ceildiv

PAGE_SIZE = 128
# Small enough to build the pool the plan describes.
BUDGET = 64 * 1024 * 1024


def _attention(window_size: int | None = None) -> MHAKVCacheParams:
    return MHAKVCacheParams(
        dtype=DType.bfloat16,
        num_layers=8,
        n_kv_heads=4,
        head_dim=128,
        enable_prefix_caching=False,
        page_size=PAGE_SIZE,
        devices=[DeviceRef.CPU()],
        window_size=window_size,
    )


def _hybrid() -> MultiKVCacheParams:
    """A global group, a sliding-window group, and conv state."""
    return MultiKVCacheParams.from_params(
        {
            "full_attention": _attention(),
            "sliding_attention": _attention(window_size=512),
            "conv_state": RecurrentStateParams(
                regions=(
                    RecurrentStateRegion(
                        leaf_id="conv/k",
                        num_layers=8,
                        row_shape=(4, 4),
                        dtype=DType.float32,
                    ),
                ),
                devices=[DeviceRef.CPU()],
            ),
        }
    )


def _uniform_cap(params: MultiKVCacheParams | MHAKVCacheParams) -> int:
    cap = compute_max_seq_len_fitting_in_cache(
        params=params, available_cache_memory=BUDGET, include_null_block=True
    )
    assert cap is not None
    return cap


def test_a_hybrid_is_capped_at_exactly_what_its_slab_admits() -> None:
    """A window and a state plateau, so the slab holds a longer request than
    uniform pages; the cap must match the pool's own fit check."""
    params = _hybrid()
    cap = JengaKVCacheManager.max_seq_len_fitting_in_cache(params, BUDGET)
    assert cap is not None
    assert cap > _uniform_cap(params)

    geometry = JengaKVCacheManager._plan_geometry(params, BUDGET)
    pool = JengaBlockPool(geometry.num_huge_blocks, geometry.ratios)

    def admits(seq_len: int) -> bool:
        num_blocks = ceildiv(seq_len, PAGE_SIZE)
        demand = {
            leaf_id: leaf.blocks_to_reserve(num_blocks)
            for leaf_id, leaf in params.leaves().items()
        }
        return pool.can_satisfy_demand(demand, at_capacity=True)

    assert admits(cap)
    assert not admits(cap + 1)

    # A state can only live on a slab, whatever the allowlist says.
    assert (
        max_seq_len_fitting_in_cache(
            params, BUDGET, is_di_enabled=True, model_name="Qwen/Qwen3-8B"
        )
        == cap
    )


def test_a_homogeneous_cache_is_capped_the_same_either_way() -> None:
    """With only full leaves both managers price a slot the same; Jenga alone
    spends one page on the null block."""
    params = _attention()
    uniform = _uniform_cap(params)
    slab = JengaKVCacheManager.max_seq_len_fitting_in_cache(params, BUDGET)
    assert slab is not None
    assert uniform - PAGE_SIZE <= slab <= uniform
