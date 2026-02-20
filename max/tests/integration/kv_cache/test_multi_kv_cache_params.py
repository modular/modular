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

"""Tests for MultiKVCacheParams memory estimation functionality."""

from __future__ import annotations

import pytest
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.legacy.kv_cache import (
    KVCacheParams,
    MultiKVCacheParams,
    compute_max_seq_len_fitting_in_cache,
    compute_num_device_blocks,
    estimated_memory_size,
)


def create_kv_cache_params(
    num_layers: int = 32,
    n_kv_heads: int = 8,
    head_dim: int = 128,
    page_size: int = 128,
    dtype: DType = DType.bfloat16,
) -> KVCacheParams:
    """Helper to create KVCacheParams with common defaults."""
    return KVCacheParams(
        dtype=dtype,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        num_layers=num_layers,
        devices=[DeviceRef.GPU()],
        cache_strategy="paged",
        page_size=page_size,
    )


class TestMultiKVCacheParamsValidation:
    """Tests for MultiKVCacheParams validation logic."""

    def test_empty_params_raises_error(self) -> None:
        """MultiKVCacheParams should raise an error if params list is empty."""
        with pytest.raises(ValueError, match="requires at least one param"):
            MultiKVCacheParams.from_params()

    def test_mismatched_cache_strategy_raises_error(self) -> None:
        """MultiKVCacheParams should raise if cache strategies don't match."""
        params1 = create_kv_cache_params()
        # Create a params with different strategy by modifying after creation
        params2 = KVCacheParams(
            dtype=DType.bfloat16,
            n_kv_heads=8,
            head_dim=128,
            num_layers=32,
            devices=[DeviceRef.GPU()],
            cache_strategy="model_default",
            page_size=128,
        )
        with pytest.raises(ValueError, match="same cache strategy"):
            MultiKVCacheParams.from_params(params1, params2)

    def test_mismatched_page_size_raises_error(self) -> None:
        """MultiKVCacheParams should raise if page sizes don't match."""
        params1 = create_kv_cache_params(page_size=128)
        params2 = create_kv_cache_params(page_size=256)
        with pytest.raises(ValueError, match="same page size"):
            MultiKVCacheParams.from_params(params1, params2)

    def test_mismatched_data_parallel_degree_raises_error(self) -> None:
        """MultiKVCacheParams should raise if data parallel degrees don't match."""
        params1 = KVCacheParams(
            dtype=DType.bfloat16,
            n_kv_heads=8,
            head_dim=128,
            num_layers=32,
            devices=[DeviceRef.GPU()],
            page_size=128,
            data_parallel_degree=1,
        )
        params2 = KVCacheParams(
            dtype=DType.bfloat16,
            n_kv_heads=8,
            head_dim=128,
            num_layers=32,
            devices=[DeviceRef.GPU(), DeviceRef.GPU()],
            page_size=128,
            data_parallel_degree=2,
        )
        with pytest.raises(ValueError, match="same data parallel degree"):
            MultiKVCacheParams.from_params(params1, params2)


class TestMultiKVCacheParamsBytesPerBlock:
    """Tests for MultiKVCacheParams.bytes_per_block aggregation."""

    def test_bytes_per_block_sums_across_params(self) -> None:
        """bytes_per_block should be the sum across all param sets."""
        params1 = create_kv_cache_params(num_layers=16, n_kv_heads=8)
        params2 = create_kv_cache_params(num_layers=16, n_kv_heads=4)

        multi_params = MultiKVCacheParams.from_params(params1, params2)

        expected = params1.bytes_per_block + params2.bytes_per_block
        assert multi_params.bytes_per_block == expected

    def test_single_param_bytes_per_block_unchanged(self) -> None:
        """With a single param, bytes_per_block should match that param."""
        params = create_kv_cache_params()
        multi_params = MultiKVCacheParams.from_params(params)

        assert multi_params.bytes_per_block == params.bytes_per_block


class TestMultiKVCacheParamsMemoryEstimation:
    """Tests for memory estimation with MultiKVCacheParams."""

    def test_compute_max_seq_len_accounts_for_all_caches(self) -> None:
        """Max sequence length should account for memory from all caches."""
        # Create two cache configs with different sizes
        params1 = create_kv_cache_params(num_layers=16, n_kv_heads=8)
        params2 = create_kv_cache_params(num_layers=16, n_kv_heads=8)

        # Available memory that can fit some blocks
        available_memory = 100 * 1024 * 1024  # 100 MB

        # Compute max seq len for individual params
        max_seq_len_1 = compute_max_seq_len_fitting_in_cache(
            params1, available_memory
        )
        max_seq_len_2 = compute_max_seq_len_fitting_in_cache(
            params2, available_memory
        )

        # Compute max seq len for multi params
        multi_params = MultiKVCacheParams.from_params(params1, params2)
        max_seq_len_multi = compute_max_seq_len_fitting_in_cache(
            multi_params, available_memory
        )

        # With two identical caches, multi should fit roughly half the seq len
        # (since bytes_per_block is doubled)
        assert max_seq_len_multi < max_seq_len_1
        assert max_seq_len_multi < max_seq_len_2

        # More precisely: since bytes_per_block doubles, seq len should halve
        # (approximately, due to integer division)
        assert max_seq_len_multi == pytest.approx(
            max_seq_len_1 / 2, rel=0.1
        ) or max_seq_len_multi == pytest.approx(max_seq_len_2 / 2, rel=0.1)

    def test_compute_num_device_blocks_with_multi_params(self) -> None:
        """compute_num_device_blocks should work correctly with MultiKVCacheParams."""
        params1 = create_kv_cache_params(num_layers=16)
        params2 = create_kv_cache_params(num_layers=16)
        multi_params = MultiKVCacheParams.from_params(params1, params2)

        available_memory = 100 * 1024 * 1024  # 100 MB

        # With multi params, we should get half the blocks (double bytes_per_block)
        blocks_single = compute_num_device_blocks(
            params1,
            available_cache_memory=available_memory,
            max_batch_size=None,
            max_seq_len=None,
        )
        blocks_multi = compute_num_device_blocks(
            multi_params,
            available_cache_memory=available_memory,
            max_batch_size=None,
            max_seq_len=None,
        )

        assert blocks_multi == blocks_single // 2

    def test_estimated_memory_size_with_multi_params(self) -> None:
        """estimated_memory_size should work correctly with MultiKVCacheParams."""
        params1 = create_kv_cache_params(num_layers=16)
        params2 = create_kv_cache_params(num_layers=16)
        multi_params = MultiKVCacheParams.from_params(params1, params2)

        available_memory = 100 * 1024 * 1024  # 100 MB
        max_batch_size = 4
        max_seq_len = 1024

        # Estimate memory for multi params
        mem_estimate = estimated_memory_size(
            multi_params,
            available_cache_memory=available_memory,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
        )

        # Memory estimate should be positive and reasonable
        assert mem_estimate > 0
        assert mem_estimate <= available_memory

    def test_multi_params_with_different_head_dims(self) -> None:
        """Test MultiKVCacheParams with different head dimensions."""
        # Simulate a model with two different cache configs (e.g., MLA + standard)
        params1 = create_kv_cache_params(
            num_layers=32, n_kv_heads=8, head_dim=128
        )
        params2 = create_kv_cache_params(
            num_layers=32, n_kv_heads=8, head_dim=64
        )

        multi_params = MultiKVCacheParams.from_params(params1, params2)

        # bytes_per_block should reflect the sum
        assert (
            multi_params.bytes_per_block
            == params1.bytes_per_block + params2.bytes_per_block
        )

        # Memory estimation should still work
        available_memory = 100 * 1024 * 1024
        max_seq_len = compute_max_seq_len_fitting_in_cache(
            multi_params, available_memory
        )
        assert max_seq_len > 0


class TestMultiKVCacheParamsProperties:
    """Tests for MultiKVCacheParams property accessors."""

    def test_properties_from_first_param(self) -> None:
        """Properties like page_size should come from the first param."""
        params1 = create_kv_cache_params(num_layers=16, page_size=128)
        params2 = create_kv_cache_params(num_layers=32, page_size=128)

        multi_params = MultiKVCacheParams.from_params(params1, params2)

        assert multi_params.page_size == 128
        assert multi_params.data_parallel_degree == 1
        assert multi_params.n_devices == 1

    def test_frozen_dataclass(self) -> None:
        """MultiKVCacheParams should be frozen (immutable)."""
        params = create_kv_cache_params()
        multi_params = MultiKVCacheParams.from_params(params)

        with pytest.raises(AttributeError):
            multi_params.params = []  # type: ignore[misc]
