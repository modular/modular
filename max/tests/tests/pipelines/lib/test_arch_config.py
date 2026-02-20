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
"""Tests for architecture-specific config interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import NonCallableMock

import pytest
from max.driver import (
    load_devices,
    scan_available_devices,
    set_virtual_device_api,
    set_virtual_device_count,
    set_virtual_device_target_arch,
)
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.legacy.kv_cache import KVCacheParams, KVCacheStrategy
from max.pipelines.lib import KVCacheConfig, MAXModelConfig, PipelineConfig
from max.pipelines.lib.config_enums import SupportedEncoding
from max.pipelines.lib.interfaces.arch_config import (
    ArchConfig,
    ArchConfigWithAttentionKVCache,
    ArchConfigWithKVCache,
)


@dataclass
class ConcreteArchConfig(ArchConfigWithAttentionKVCache):
    """Concrete implementation of ArchConfigWithAttentionKVCache for testing."""

    # Required attributes can be implemented as dataclass fields.
    num_key_value_heads: int = 8
    head_dim: int = 64
    model_max_seq_len: int = 2048

    # Or as properties.
    @property
    def num_layers(self) -> int:
        return 12


def create_mock_pipeline_config(
    quantization_encoding: SupportedEncoding | None = "bfloat16",
    kv_cache_page_size: int = 128,
    cache_strategy: KVCacheStrategy = "paged",
    enable_prefix_caching: bool = True,
    enable_kvcache_swapping_to_host: bool = False,
    host_kvcache_swap_space_gb: float = 50.0,
    data_parallel_degree: int = 1,
    max_length: int | None = None,
) -> NonCallableMock:
    """Create a mock PipelineConfig for testing."""
    mock_config = NonCallableMock(spec=PipelineConfig)

    # Create mock model config
    mock_model = NonCallableMock(spec=MAXModelConfig)
    mock_model.quantization_encoding = quantization_encoding
    mock_model.data_parallel_degree = data_parallel_degree
    mock_model.device_specs = []
    mock_model.max_length = max_length

    # Create mock kv_cache_config
    mock_kv_cache_config = NonCallableMock(spec=KVCacheConfig)
    mock_kv_cache_config.kv_cache_page_size = kv_cache_page_size
    mock_kv_cache_config.cache_strategy = cache_strategy
    mock_kv_cache_config.enable_prefix_caching = enable_prefix_caching
    mock_kv_cache_config.enable_kvcache_swapping_to_host = (
        enable_kvcache_swapping_to_host
    )
    mock_kv_cache_config.host_kvcache_swap_space_gb = host_kvcache_swap_space_gb
    mock_kv_cache_config.cache_dtype = DType.bfloat16

    mock_model.kv_cache = mock_kv_cache_config
    mock_config.model = mock_model

    return mock_config


@pytest.fixture(autouse=True)
def mock_default_devices() -> list[DeviceRef]:
    set_virtual_device_api("cuda")
    set_virtual_device_target_arch("sm_80")
    set_virtual_device_count(2)
    return [
        DeviceRef.from_device(device)
        for device in load_devices(scan_available_devices())
    ]


def test_arch_config_protocol_check() -> None:
    """Test that ArchConfig is a runtime-checkable protocol."""

    # A class implementing initialize method should satisfy the protocol
    class TestConfig:
        @classmethod
        def initialize(cls, pipeline_config: PipelineConfig) -> TestConfig:
            return cls()

        def get_max_seq_len(self) -> int:
            return 2048

    assert isinstance(TestConfig(), ArchConfig)


def test_arch_config_with_cache_protocol_check() -> None:
    """Test that ArchConfigWithKVCache protocol requires get_kv_params method."""
    mock_kv_params = NonCallableMock(spec=KVCacheParams)

    class TestConfigWithCache:
        @classmethod
        def initialize(
            cls, pipeline_config: PipelineConfig
        ) -> TestConfigWithCache:
            return cls()

        def get_kv_params(self) -> KVCacheParams:
            return mock_kv_params

        def get_max_seq_len(self) -> int:
            return 2048

    instance = TestConfigWithCache()
    assert isinstance(instance, ArchConfigWithKVCache)


class TestArchConfigWithAttentionKVCache:
    """Tests for ArchConfigWithAttentionKVCache."""

    def test_initialize_raises_error_when_quantization_encoding_is_none(
        self,
    ) -> None:
        """Test that initialize raises ValueError when quantization_encoding is None."""
        mock_config = create_mock_pipeline_config(quantization_encoding=None)

        with pytest.raises(
            ValueError,
            match="Quantization encoding is required for ArchConfigWithAttentionKVCache",
        ):
            ConcreteArchConfig.initialize(mock_config)

    def test_initialize_succeeds_with_valid_quantization_encoding(self) -> None:
        """Test that initialize succeeds with valid quantization encoding."""
        mock_config = create_mock_pipeline_config(
            quantization_encoding="bfloat16"
        )
        result = ConcreteArchConfig.initialize(mock_config)
        assert isinstance(result, ConcreteArchConfig)
        assert result.dtype == DType.bfloat16
        assert result.cache_dtype == DType.bfloat16
        assert result.data_parallel_degree == 1

        # Test with encoding that maps to different dtype/config

        mock_config = create_mock_pipeline_config(quantization_encoding="q4_k")
        result = ConcreteArchConfig.initialize(mock_config)
        assert result.dtype == DType.uint8
        assert result.cache_dtype == DType.bfloat16
        assert result.data_parallel_degree == 1

    def test_create_with_only_dtype(self) -> None:
        """Test that config can be created with only dtype specified."""
        config = ConcreteArchConfig(dtype=DType.bfloat16, devices=[])

        assert config.dtype == DType.bfloat16
        # devices should be what we passed
        assert config.devices == []
        # kv_cache_config should be default
        assert isinstance(config.kv_cache, KVCacheConfig)
        # data_parallel_degree should be 1
        assert config.data_parallel_degree == 1

    def test_create_with_explicit_cache_dtype(self) -> None:
        """Test that cache_dtype can be explicitly set different from dtype."""
        config = ConcreteArchConfig(
            dtype=DType.float8_e4m3fn,
            cache_dtype=DType.bfloat16,
            devices=[],
        )

        assert config.dtype == DType.float8_e4m3fn
        assert config.cache_dtype == DType.bfloat16

    def test_create_with_custom_kv_cache_config(self) -> None:
        """Test that custom kv_cache_config can be provided."""
        custom_kv_config = KVCacheConfig(
            kv_cache_page_size=256,
            enable_prefix_caching=False,
        )

        config = ConcreteArchConfig(
            dtype=DType.bfloat16,
            kv_cache=custom_kv_config,
            devices=[],
        )

        assert config.kv_cache.kv_cache_page_size == 256
        assert config.kv_cache.enable_prefix_caching is False

    def test_default_devices_uses_factory(
        self, mock_default_devices: list[DeviceRef]
    ) -> None:
        """Test that devices defaults to scanning available devices."""
        config = ConcreteArchConfig(dtype=DType.bfloat16)

        # Should use the devices from the autouse fixture
        assert config.devices == mock_default_devices
        assert len(config.devices) == 2

    def test_abstract_properties_from_subclass(self) -> None:
        """Test that abstract properties are correctly implemented in subclass."""
        config = ConcreteArchConfig(dtype=DType.bfloat16, devices=[])

        assert config.num_key_value_heads == 8
        assert config.head_dim == 64
        assert config.num_layers == 12
        assert config.model_max_seq_len == 2048

    def test_get_kv_params_returns_correct_kv_cache_params(
        self, mock_default_devices: list[DeviceRef]
    ) -> None:
        """Test that get_kv_params method correctly constructs KVCacheParams."""
        custom_kv_config = KVCacheConfig(
            kv_cache_page_size=256,
            cache_strategy="paged",
            enable_prefix_caching=True,
            enable_kvcache_swapping_to_host=True,
            host_kvcache_swap_space_gb=100.0,
        )

        config = ConcreteArchConfig(
            dtype=DType.bfloat16,
            kv_cache=custom_kv_config,
            data_parallel_degree=2,
            devices=mock_default_devices,
        )
        kv_params = config.get_kv_params()

        # Verify the KVCacheParams fields
        assert kv_params.dtype == DType.bfloat16
        assert kv_params.n_kv_heads == 8  # from ConcreteArchConfig
        assert kv_params.head_dim == 64  # from ConcreteArchConfig
        assert kv_params.num_layers == 12  # from ConcreteArchConfig
        assert kv_params.page_size == 256
        assert kv_params.cache_strategy == "paged"
        assert kv_params.enable_prefix_caching is True
        assert kv_params.enable_kvcache_swapping_to_host is True
        assert kv_params.host_kvcache_swap_space_gb == 100.0
        assert kv_params.data_parallel_degree == 2

        # Test that kv params are cached.
        kv_params_2 = config.get_kv_params()
        assert kv_params is kv_params_2

    def test_model_equality(self) -> None:
        """Test that two configs with same values are equal."""
        config1 = ConcreteArchConfig(dtype=DType.bfloat16, devices=[])
        config2 = ConcreteArchConfig(dtype=DType.bfloat16, devices=[])

        assert config1 == config2

    def test_model_inequality(self) -> None:
        """Test that configs with different values are not equal."""
        config1 = ConcreteArchConfig(dtype=DType.bfloat16, devices=[])
        config2 = ConcreteArchConfig(dtype=DType.float32, devices=[])

        assert config1 != config2

    def test_explicit_cache_dtype_not_overwritten(self) -> None:
        """Test that explicitly set cache_dtype is not overwritten by default_factory."""
        config = ConcreteArchConfig(
            dtype=DType.float32,
            cache_dtype=DType.bfloat16,
            devices=[],
        )

        # cache_dtype should remain as explicitly set
        assert config.cache_dtype == DType.bfloat16

    def test_get_max_seq_len_defaults_to_model_max_seq_len(self) -> None:
        """Test that get_max_seq_len returns model_max_seq_len when max_length is None."""
        config = ConcreteArchConfig(dtype=DType.bfloat16, devices=[])

        # max_length is None by default, so get_max_seq_len should equal model_max_seq_len
        assert config.user_provided_max_length is None
        assert config.get_max_seq_len() == 2048

    def test_get_max_seq_len_uses_max_length_when_set(self) -> None:
        """Test that get_max_seq_len returns max_length when explicitly set."""
        config = ConcreteArchConfig(
            dtype=DType.bfloat16,
            devices=[],
            user_provided_max_length=1024,
        )

        assert config.get_max_seq_len() == 1024

    def test_get_max_seq_len_raises_when_max_length_exceeds_default(
        self,
    ) -> None:
        """Test that get_max_seq_len raises ValueError when max_length exceeds model_max_seq_len."""
        config = ConcreteArchConfig(
            dtype=DType.bfloat16,
            devices=[],
            user_provided_max_length=4096,  # Exceeds model_max_seq_len of 2048
        )

        with pytest.raises(
            ValueError,
            match=r"default value provided \(4096\) exceeds the upper bound \(2048\)",
        ):
            _ = config.get_max_seq_len()
