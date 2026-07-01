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

"""Tests for KV cache registry functions: load_kv_manager."""

from __future__ import annotations

from unittest.mock import MagicMock, Mock, patch

import pytest
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import KVCacheParams, MHAKVCacheParams
from max.pipelines.kv_cache import load_kv_manager


def create_kv_params(
    num_layers: int = 32,
    n_kv_heads: int = 8,
    head_dim: int = 128,
    page_size: int = 128,
    dtype: DType = DType.bfloat16,
) -> KVCacheParams:
    """Helper to create KVCacheParams with common defaults."""
    return MHAKVCacheParams(
        dtype=dtype,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        num_layers=num_layers,
        devices=[DeviceRef.GPU()],
        page_size=page_size,
    )


class TestLoadKvManager:
    """Tests for load_kv_manager function."""

    @patch("max.pipelines.kv_cache.registry.PagedKVCacheManager")
    def test_load_kv_manager_creates_manager(
        self, mock_paged_manager_cls: MagicMock
    ) -> None:
        """load_kv_manager should create a PagedKVCacheManager."""
        mock_manager = MagicMock()
        mock_paged_manager_cls.return_value = mock_manager

        params = create_kv_params()
        mock_session = MagicMock()

        result = load_kv_manager(
            params=params,
            max_batch_size=16,
            max_seq_len=2048,
            session=mock_session,
            available_cache_memory=1024 * 1024 * 1024,  # 1 GB
        )

        assert result == mock_manager
        mock_paged_manager_cls.assert_called_once()

    @patch("max.pipelines.kv_cache.registry.PagedKVCacheManager")
    def test_load_kv_manager_passes_correct_params(
        self, mock_paged_manager_cls: MagicMock
    ) -> None:
        """load_kv_manager should pass correct params to PagedKVCacheManager."""
        params = create_kv_params(num_layers=16)
        mock_session = MagicMock()

        load_kv_manager(
            params=params,
            max_batch_size=8,
            max_seq_len=1024,
            session=mock_session,
            available_cache_memory=512 * 1024 * 1024,  # 512 MB
        )

        call_kwargs = mock_paged_manager_cls.call_args.kwargs
        assert call_kwargs["params"] == params
        assert call_kwargs["session"] == mock_session
        assert call_kwargs["total_num_pages"] > 0

    def test_load_kv_manager_rejects_zero_batch_size(self) -> None:
        """load_kv_manager should raise ValueError for batch_size <= 0."""
        params = create_kv_params()
        mock_session = MagicMock()

        with pytest.raises(
            ValueError, match="max_batch_size must be greater than 0"
        ):
            load_kv_manager(
                params=params,
                max_batch_size=0,
                max_seq_len=2048,
                session=mock_session,
                available_cache_memory=1024 * 1024 * 1024,
            )

    def test_load_kv_manager_rejects_negative_batch_size(self) -> None:
        """load_kv_manager should raise ValueError for negative batch_size."""
        params = create_kv_params()
        mock_session = MagicMock()

        with pytest.raises(
            ValueError, match="max_batch_size must be greater than 0"
        ):
            load_kv_manager(
                params=params,
                max_batch_size=-1,
                max_seq_len=2048,
                session=mock_session,
                available_cache_memory=1024 * 1024 * 1024,
            )

    @patch("max.pipelines.kv_cache.registry.PagedKVCacheManager")
    def test_load_kv_manager_rejects_invalid_page_size(
        self, mock_paged_manager_cls: MagicMock
    ) -> None:
        """load_kv_manager should reject page sizes that aren't multiples of 128."""
        # Create params with invalid page size (not multiple of 128)
        params = MHAKVCacheParams(
            dtype=DType.bfloat16,
            n_kv_heads=8,
            head_dim=128,
            num_layers=32,
            devices=[DeviceRef.GPU()],
            page_size=64,  # Invalid: not a multiple of 128
        )
        mock_session = MagicMock()

        with pytest.raises(ValueError, match="multiple of 128"):
            load_kv_manager(
                params=params,
                max_batch_size=16,
                max_seq_len=2048,
                session=mock_session,
                available_cache_memory=1024 * 1024 * 1024,
            )


class TestLoadKvManagers:
    """Tests for load_kv_managers function (plural - supports MultiKVCacheParams)."""

    @patch("max.pipelines.kv_cache.registry.PagedKVCacheManager")
    def test_load_kv_managers_single_params(
        self, mock_paged_manager_cls: MagicMock
    ) -> None:
        """load_kv_managers should return a list with one manager for KVCacheParams."""
        mock_manager = MagicMock()
        mock_paged_manager_cls.return_value = mock_manager

        params = create_kv_params()
        mock_session = MagicMock()

        result = load_kv_manager(
            params=params,
            max_batch_size=16,
            max_seq_len=2048,
            session=mock_session,
            available_cache_memory=1024 * 1024 * 1024,
        )

        assert result == mock_manager


class TestLoadKvManagerVirtualDevice:
    """Tests for virtual device mode behavior."""

    @patch(
        "max.pipelines.kv_cache.registry.is_virtual_device_mode",
        return_value=True,
    )
    def test_load_kv_manager_returns_mock_in_virtual_mode(
        self, mock_is_virtual: MagicMock
    ) -> None:
        """In virtual device mode, load_kv_manager should return a Mock."""
        params = create_kv_params()
        mock_session = MagicMock()

        result = load_kv_manager(
            params=params,
            max_batch_size=16,
            max_seq_len=2048,
            session=mock_session,
            available_cache_memory=1024 * 1024 * 1024,
        )

        assert isinstance(result, Mock)
