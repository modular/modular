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

from dataclasses import replace
from unittest.mock import MagicMock, Mock, patch

import pytest
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.nn.kv_cache import (
    KVCacheParams,
    KVCacheQuantizationConfig,
    KVConnectorType,
    MHAKVCacheParams,
    MLAKVCacheParams,
    MultiKVCacheParams,
    NullKVConnectorConfig,
    RecurrentStateParams,
    RecurrentStateRegion,
    estimated_memory_size,
)
from max.nn.kv_cache.cache_params import TIERED_CONNECTOR_TYPES
from max.pipelines.kv_cache import (
    JengaKVCacheManager,
    PagedKVCacheManager,
    PagedKVCacheManagerInterface,
    kv_cache_memory_size,
    load_kv_manager,
)
from max.pipelines.kv_cache.config import KVConnectorConfig
from max.pipelines.kv_cache.registry import (
    _use_jenga_kv_cache,
    jenga_supports_di,
)


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


def _load_kv_manager_with_defaults(
    params: KVCacheParams,
    max_batch_size: int,
    max_seq_len: int,
    session: InferenceSession,
    available_cache_memory: int,
    is_di_enabled: bool = False,
    model_name: str = "FAKE",
) -> PagedKVCacheManagerInterface:
    return load_kv_manager(
        params=params,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        session=session,
        available_cache_memory=available_cache_memory,
        is_di_enabled=is_di_enabled,
        model_name=model_name,
        max_num_input_tokens=None,
    )


class TestUseJengaKvCache:
    """Default and opt-out behavior for the Jenga manager."""

    @pytest.mark.parametrize(
        "model_name",
        ["openai/gpt-oss-20b", "Qwen/Qwen3-8B", "poolside/Laguna-S-2.1-NVFP4"],
    )
    def test_legacy_env_disables_jenga(
        self, model_name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MODULAR_USE_LEGACY_KV_CACHE", "1")
        assert not _use_jenga_kv_cache(
            create_kv_params(),
            is_di_enabled=False,
            model_name=model_name,
        )

    def test_disaggregated_inference_keeps_jenga_for_full_attention(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MODULAR_USE_LEGACY_KV_CACHE", raising=False)
        assert _use_jenga_kv_cache(
            create_kv_params(),
            is_di_enabled=True,
            model_name="meta-llama/Llama-3.1-8B-Instruct",
        )

    def test_disaggregated_inference_rejects_sliding_window(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MODULAR_USE_LEGACY_KV_CACHE", raising=False)
        sliding = MHAKVCacheParams(
            dtype=DType.bfloat16,
            n_kv_heads=8,
            head_dim=128,
            num_layers=2,
            devices=[DeviceRef.GPU()],
            page_size=128,
            window_size=4096,
        )
        assert not jenga_supports_di(sliding)
        with pytest.raises(ValueError, match="only full-attention"):
            _use_jenga_kv_cache(
                sliding,
                is_di_enabled=True,
                model_name="openai/gpt-oss-20b",
            )

    def test_legacy_env_rejects_disaggregated_inference(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MODULAR_USE_LEGACY_KV_CACHE", "1")
        with pytest.raises(
            ValueError,
            match=r"Disaggregated inference.*legacy KV manager.*Unset"
            r" MODULAR_USE_LEGACY_KV_CACHE",
        ):
            _use_jenga_kv_cache(
                create_kv_params(),
                is_di_enabled=True,
                model_name="meta-llama/Llama-3.1-8B-Instruct",
            )

    @pytest.mark.parametrize(
        "connector",
        [
            KVConnectorType.tiered,
            KVConnectorType.rust_tiered,
            KVConnectorType.dkv,
        ],
    )
    def test_legacy_env_rejects_kv_connector(
        self, connector: KVConnectorType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MODULAR_USE_LEGACY_KV_CACHE", "1")
        params = replace(
            create_kv_params(),
            enable_prefix_caching=True,
            kv_connector_config=NullKVConnectorConfig(type=connector),
        )
        with pytest.raises(
            ValueError,
            match=rf"'{connector.value}' KV connector.*legacy KV manager.*"
            r"Unset MODULAR_USE_LEGACY_KV_CACHE",
        ):
            _use_jenga_kv_cache(params, is_di_enabled=False, model_name="FAKE")

    def test_jenga_accepts_kv_connector(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MODULAR_USE_LEGACY_KV_CACHE", raising=False)
        params = replace(
            create_kv_params(),
            enable_prefix_caching=True,
            kv_connector_config=NullKVConnectorConfig(
                type=KVConnectorType.rust_tiered
            ),
        )
        assert _use_jenga_kv_cache(
            params, is_di_enabled=False, model_name="FAKE"
        )

    def test_a_state_selects_jenga_whatever_else_says(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A recurrent state can only live on Jenga."""
        monkeypatch.setenv("MODULAR_USE_LEGACY_KV_CACHE", "1")
        attn = create_kv_params()
        with_state = MultiKVCacheParams.from_params(
            {
                "attn": attn,
                "state": RecurrentStateParams(
                    regions=(
                        RecurrentStateRegion(
                            leaf_id="conv_state",
                            num_layers=2,
                            row_shape=(8, 3),
                            dtype=DType.float32,
                        ),
                    ),
                    devices=attn.devices,
                ),
            }
        )
        assert _use_jenga_kv_cache(
            with_state, is_di_enabled=False, model_name="FAKE"
        )
        with pytest.raises(ValueError, match="only full-attention"):
            _use_jenga_kv_cache(
                with_state, is_di_enabled=True, model_name="FAKE"
            )

    def test_legacy_env_rejects_scratch_leaf(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MODULAR_USE_LEGACY_KV_CACHE", "1")
        params = MultiKVCacheParams.from_params(
            {
                "attn": create_kv_params(),
                "scratch": replace(
                    create_kv_params(num_layers=2), scratch=True
                ),
            }
        )
        with pytest.raises(
            ValueError,
            match=r"scratch KV cache leaf.*Unset MODULAR_USE_LEGACY_KV_CACHE",
        ):
            _use_jenga_kv_cache(params, is_di_enabled=False, model_name="FAKE")

    def test_nvfp4_mla_cache_selects_jenga(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An NVFP4 MLA cache pages its scales as their own Jenga leaf. The
        sparse MLA kernels read them through the scales lookup table, so it
        needs no legacy cache, which has no KV connectors or disaggregated
        inference."""
        monkeypatch.delenv("MODULAR_USE_LEGACY_KV_CACHE", raising=False)
        nvfp4 = MLAKVCacheParams(
            dtype=DType.uint8,
            head_dim=576,
            num_layers=2,
            devices=[DeviceRef.GPU()],
            page_size=128,
            num_q_heads=8,
            kvcache_quant_config=KVCacheQuantizationConfig(
                scale_dtype=DType.float8_e4m3fn,
                quantization_granularity=16,
                fp8_tail_dim=64,
            ),
        )
        assert nvfp4.is_nvfp4_kv_cache
        multi = MultiKVCacheParams.from_params(
            {"mla": nvfp4, "indexer": create_kv_params(num_layers=2)}
        )
        for params in (nvfp4, multi):
            for is_di_enabled in (False, True):
                assert _use_jenga_kv_cache(
                    params, is_di_enabled=is_di_enabled, model_name="FAKE"
                )


@pytest.mark.parametrize(
    "model_name", ["Qwen/Qwen3-8B", "FAKE", "/local/model"]
)
@patch("max.pipelines.kv_cache.registry.JengaKVCacheManager.create")
def test_load_kv_manager_defaults_to_jenga(
    mock_create: MagicMock,
    model_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MODULAR_USE_LEGACY_KV_CACHE", raising=False)
    params = create_kv_params()
    result = _load_kv_manager_with_defaults(
        params=params,
        max_batch_size=16,
        max_seq_len=2048,
        session=MagicMock(),
        available_cache_memory=1024**3,
        model_name=model_name,
    )
    assert result is mock_create.return_value
    mock_create.assert_called_once_with(
        params=params,
        available_bytes=1024**3,
        max_batch_size=16,
        max_num_input_tokens=None,
        max_seq_len=2048,
    )


@pytest.fixture
def legacy_kv_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicitly select the legacy manager for its allocation tests."""
    monkeypatch.setenv("MODULAR_USE_LEGACY_KV_CACHE", "1")


@pytest.mark.usefixtures("legacy_kv_cache")
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

        result = _load_kv_manager_with_defaults(
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

        _load_kv_manager_with_defaults(
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

    @patch("max.pipelines.kv_cache.registry.PagedKVCacheManager")
    def test_load_kv_manager_fits_null_block_in_budget(
        self, mock_paged_manager_cls: MagicMock
    ) -> None:
        """The pages plus the manager's extra null block fit the budget.

        Memory estimation can set the budget to the device's largest single
        allocation, so a page past it fails to allocate. The batch asks for
        more pages than the budget holds, so the budget sets the page count.
        """
        params = create_kv_params(num_layers=4)
        budget = 100 * params.bytes_per_block

        _load_kv_manager_with_defaults(
            params=params,
            max_batch_size=1000,
            max_seq_len=params.page_size,
            session=MagicMock(),
            available_cache_memory=budget,
        )

        total_num_pages = mock_paged_manager_cls.call_args.kwargs[
            "total_num_pages"
        ]
        assert (total_num_pages + 1) * params.bytes_per_block <= budget

    def test_load_kv_manager_rejects_zero_batch_size(self) -> None:
        """load_kv_manager should raise ValueError for batch_size <= 0."""
        params = create_kv_params()
        mock_session = MagicMock()

        with pytest.raises(
            ValueError, match="max_batch_size must be greater than 0"
        ):
            _load_kv_manager_with_defaults(
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
            _load_kv_manager_with_defaults(
                params=params,
                max_batch_size=-1,
                max_seq_len=2048,
                session=mock_session,
                available_cache_memory=1024 * 1024 * 1024,
            )

    def test_load_kv_manager_rejects_oversized_max_seq_len(self) -> None:
        """load_kv_manager should fail startup when a single request at
        max_seq_len cannot fit in the device block pool."""
        params = create_kv_params()
        mock_session = MagicMock()

        # 1 GiB fits 64 blocks of 128 tokens each = 8192 tokens.
        with pytest.raises(
            RuntimeError, match="one request at the max sequence length"
        ):
            _load_kv_manager_with_defaults(
                params=params,
                max_batch_size=16,
                max_seq_len=8192 + 1,
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
            _load_kv_manager_with_defaults(
                params=params,
                max_batch_size=16,
                max_seq_len=2048,
                session=mock_session,
                available_cache_memory=1024 * 1024 * 1024,
            )


@pytest.mark.usefixtures("legacy_kv_cache")
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

        result = _load_kv_manager_with_defaults(
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

        result = _load_kv_manager_with_defaults(
            params=params,
            max_batch_size=16,
            max_seq_len=2048,
            session=mock_session,
            available_cache_memory=1024 * 1024 * 1024,
        )

        assert isinstance(result, Mock)


class TestKvCacheMemorySize:
    """Tests the KV budget a pipeline's memory plan reserves."""

    PLENTY = 256 * 1024**3
    BATCH = 8
    SEQ = 8192

    def _windowed(self) -> MultiKVCacheParams:
        """A sliding-window leaf beside a full one, as Gemma declares them."""
        full = create_kv_params(num_layers=4)
        sliding = MHAKVCacheParams(
            dtype=DType.bfloat16,
            n_kv_heads=8,
            head_dim=128,
            num_layers=20,
            devices=full.devices,
            page_size=128,
            window_size=1024,
        )
        return MultiKVCacheParams.from_params(
            {"full": full, "sliding": sliding}
        )

    def _size(self, params: MultiKVCacheParams, model_name: str) -> int:
        return kv_cache_memory_size(
            params,
            self.PLENTY,
            self.BATCH,
            self.SEQ,
            is_di_enabled=False,
            model_name=model_name,
        )

    def test_a_windowed_jenga_cache_keeps_the_generic_estimate(self) -> None:
        """Checks Jenga never sizes a windowed cache below the generic estimate.

        The generic estimate charges a windowed leaf for every token, and
        that room holds cached prefixes, so the exact need alone would take
        it away.
        """
        params = self._windowed()
        generic = estimated_memory_size(
            params=params,
            available_cache_memory=self.PLENTY,
            max_batch_size=self.BATCH,
            max_seq_len=self.SEQ,
            include_null_block=True,
        )
        exact = JengaKVCacheManager.memory_size(
            params, self.PLENTY, self.BATCH, self.SEQ
        )
        assert exact < generic, "the fixture must over-count its window"

        assert self._size(params, "google/gemma-4-31B-it") == generic

    def test_a_state_cache_gets_at_least_its_exact_need(self) -> None:
        """Checks a state the generic estimate does not price still fits."""
        attn = create_kv_params()
        params = MultiKVCacheParams.from_params(
            {
                "attn": attn,
                "state": RecurrentStateParams(
                    regions=(
                        RecurrentStateRegion(
                            leaf_id="conv_state",
                            num_layers=64,
                            row_shape=(4096, 3),
                            dtype=DType.float32,
                        ),
                    ),
                    devices=attn.devices,
                ),
            }
        )
        exact = JengaKVCacheManager.memory_size(
            params, self.PLENTY, self.BATCH, self.SEQ
        )

        assert self._size(params, "FAKE") >= exact


@pytest.mark.parametrize(
    "connector",
    [
        KVConnectorType.tiered,
        KVConnectorType.rust_tiered,
        KVConnectorType.dkv,
    ],
)
def test_legacy_manager_rejects_kv_connector(
    connector: KVConnectorType,
) -> None:
    params = replace(
        create_kv_params(),
        enable_prefix_caching=True,
        kv_connector_config=NullKVConnectorConfig(type=connector),
    )
    with pytest.raises(ValueError, match="incompatible with the legacy"):
        PagedKVCacheManager(
            params=params,
            session=MagicMock(),
            total_num_pages=8,
            max_batch_size=1,
        )


def test_legacy_manager_rejects_scratch_leaf() -> None:
    params = replace(create_kv_params(), scratch=True)
    with pytest.raises(ValueError, match="cannot hold a scratch leaf"):
        PagedKVCacheManager(
            params=params,
            session=MagicMock(),
            total_num_pages=8,
            max_batch_size=1,
        )


def test_tiered_connector_types() -> None:
    assert TIERED_CONNECTOR_TYPES == {
        KVConnectorType.tiered,
        KVConnectorType.rust_tiered,
    }


def test_kv_connector_type_rejects_removed_mojo_tiered() -> None:
    with pytest.raises(
        ValueError,
        match=r"'mojo_tiered' KV connector was removed.*rust_tiered",
    ):
        KVConnectorType("mojo_tiered")
    with pytest.raises(ValueError, match="not a valid KVConnectorType"):
        KVConnectorType("bogus")


def test_kv_connector_config_rejects_removed_mojo_tiered() -> None:
    with pytest.raises(
        ValueError,
        match=r"'mojo_tiered' KV connector was removed.*rust_tiered",
    ):
        KVConnectorConfig.model_validate({"type": "mojo_tiered"})


def test_kv_connector_config_typo_error_lists_the_choices() -> None:
    with pytest.raises(ValueError, match=r"rust_tiered.*dkv"):
        KVConnectorConfig.model_validate({"type": "rust_tierd"})


def test_kv_connector_config_stores_the_enum() -> None:
    """Plain strings from a config file or the CLI are stored as the enum."""
    assert (
        KVConnectorConfig.model_validate({"type": "rust_tiered"}).type
        is KVConnectorType.rust_tiered
    )
    assert (
        KVConnectorConfig.model_validate_json('{"type": "dkv"}').type
        is KVConnectorType.dkv
    )
