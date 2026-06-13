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

import numpy as np
import pytest
from max.driver import CPU
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import (
    AttnKey,
    BatchCharacteristics,
    KVCacheParams,
    KVCacheQuantizationConfig,
    KVConnectorType,
    MHAAttnKey,
    MLAAttnKey,
)


def test_single_device_compatible() -> None:
    """Test single device configuration (no DP or TP)."""
    params = KVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=8,
        head_dim=128,
        num_layers=1,
        devices=[DeviceRef.GPU()],
        data_parallel_degree=1,
        page_size=16,
    )
    assert params.n_kv_heads_per_device == 8


def test_tensor_parallel_compatible_divisible_heads() -> None:
    """Test TP mode with n_kv_heads divisible by n_devices."""
    params = KVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=8,
        head_dim=128,
        num_layers=1,
        devices=[DeviceRef.GPU(i) for i in range(2)],
        data_parallel_degree=1,
        page_size=16,
    )
    assert params.n_kv_heads_per_device == 4


def test_tensor_parallel_compatible_multiple_devices() -> None:
    """Test TP mode with 4 devices and 16 heads."""
    params = KVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=16,
        head_dim=128,
        num_layers=1,
        devices=[DeviceRef.GPU(i) for i in range(4)],
        data_parallel_degree=1,
        page_size=16,
    )
    assert params.n_kv_heads_per_device == 4


def test_tensor_parallel_compatible_large_heads() -> None:
    """Test TP mode with many heads evenly distributed."""
    params = KVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=32,
        head_dim=128,
        num_layers=1,
        devices=[DeviceRef.GPU(i) for i in range(8)],
        data_parallel_degree=1,
        page_size=16,
    )
    assert params.n_kv_heads_per_device == 4


def test_data_parallel_compatible_equal_devices() -> None:
    """Test DP mode with data_parallel_degree equal to n_devices."""
    params = KVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=8,
        head_dim=128,
        num_layers=1,
        devices=[DeviceRef.GPU(i) for i in range(4)],
        data_parallel_degree=4,
        page_size=16,
    )
    # In DP mode, heads are not sharded
    assert params.n_kv_heads_per_device == 8


def test_data_parallel_compatible_multiple_devices() -> None:
    """Test DP mode with multiple devices."""
    params = KVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=12,
        head_dim=64,
        num_layers=1,
        devices=[DeviceRef.GPU(i) for i in range(2)],
        data_parallel_degree=2,
        page_size=16,
    )
    # In DP mode, all heads are on each device
    assert params.n_kv_heads_per_device == 12


# ==================== Incompatible Cases ====================


def test_data_parallel_exceeds_devices_fails() -> None:
    """Test that DP degree > n_devices raises ValueError."""
    with pytest.raises(
        ValueError,
        match=r"Data parallelism degree \(4\) cannot be greater than the number of devices \(2\)",
    ):
        KVCacheParams(
            dtype=DType.bfloat16,
            n_kv_heads=8,
            head_dim=128,
            num_layers=1,
            devices=[DeviceRef.GPU(i) for i in range(2)],
            data_parallel_degree=4,
            page_size=16,
        )


def test_data_parallel_exceeds_devices_large_degree_fails() -> None:
    """Test that DP degree >> n_devices raises ValueError."""
    with pytest.raises(
        ValueError,
        match=r"Data parallelism degree \(8\) cannot be greater than the number of devices \(1\)",
    ):
        KVCacheParams(
            dtype=DType.bfloat16,
            n_kv_heads=16,
            head_dim=128,
            num_layers=1,
            devices=[DeviceRef.GPU()],
            data_parallel_degree=8,
            page_size=16,
        )


def test_mixed_dp_tp_not_supported_fails() -> None:
    """Test that DP + TP combination is not yet supported."""
    with pytest.raises(
        ValueError,
        match=r"We do not yet support DP \+ TP at the same time.*data_parallel_degree=2.*n_devices=4",
    ):
        KVCacheParams(
            dtype=DType.bfloat16,
            n_kv_heads=8,
            head_dim=128,
            num_layers=1,
            devices=[DeviceRef.GPU(i) for i in range(4)],
            data_parallel_degree=2,
            page_size=16,
        )


def test_mixed_dp_tp_another_combination_fails() -> None:
    """Test another DP + TP combination that should fail."""
    with pytest.raises(
        ValueError,
        match=r"We do not yet support DP \+ TP at the same time.*data_parallel_degree=3.*n_devices=6",
    ):
        KVCacheParams(
            dtype=DType.bfloat16,
            n_kv_heads=12,
            head_dim=64,
            num_layers=1,
            devices=[DeviceRef.GPU(i) for i in range(6)],
            data_parallel_degree=3,
            page_size=16,
        )


def test_tensor_parallel_non_divisible_heads_fails() -> None:
    """Test that TP mode with non-divisible heads raises ValueError."""
    with pytest.raises(
        ValueError,
        match=r"Number of KV heads \(8\) must be divisible by the number of devices \(3\)",
    ):
        KVCacheParams(
            dtype=DType.bfloat16,
            n_kv_heads=8,
            head_dim=128,
            num_layers=1,
            devices=[DeviceRef.GPU(i) for i in range(3)],
            data_parallel_degree=1,
            page_size=16,
        )


def test_tensor_parallel_non_divisible_heads_small_fails() -> None:
    """Test TP mode where n_kv_heads < n_devices."""
    with pytest.raises(
        ValueError,
        match=r"Number of KV heads \(2\) must be divisible by the number of devices \(4\)",
    ):
        KVCacheParams(
            dtype=DType.bfloat16,
            n_kv_heads=2,
            head_dim=128,
            num_layers=1,
            devices=[DeviceRef.GPU(i) for i in range(4)],
            data_parallel_degree=1,
            page_size=16,
        )


def test_tensor_parallel_odd_division_fails() -> None:
    """Test TP mode with an odd number that doesn't divide evenly."""
    with pytest.raises(
        ValueError,
        match=r"Number of KV heads \(7\) must be divisible by the number of devices \(2\)",
    ):
        KVCacheParams(
            dtype=DType.bfloat16,
            n_kv_heads=7,
            head_dim=128,
            num_layers=1,
            devices=[DeviceRef.GPU(i) for i in range(2)],
            data_parallel_degree=1,
            page_size=16,
        )


# ==================== copy_as_dp_1 Tests ====================


def test_copy_as_dp_1_basic() -> None:
    """Test copy_as_dp_1 creates a new instance with DP=1."""
    params = KVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=8,
        head_dim=128,
        num_layers=1,
        devices=[DeviceRef.GPU(i) for i in range(4)],
        data_parallel_degree=4,
        page_size=16,
        enable_prefix_caching=True,
    )

    copied = params.copy_as_dp_1()

    # Verify DP is set to 1
    assert copied.data_parallel_degree == 1
    # Verify n_devices is adjusted correctly
    assert copied.n_devices == 1
    # Verify n_kv_heads_per_device is recomputed correctly (TP mode now)
    assert copied.n_kv_heads_per_device == 8
    # Verify other parameters are preserved
    assert copied.dtype == DType.bfloat16
    assert copied.n_kv_heads == 8
    assert copied.head_dim == 128
    assert copied.page_size == 16
    assert copied.enable_prefix_caching is True


def test_copy_as_dp_1_preserves_all_parameters() -> None:
    """Test that copy_as_dp_1 preserves all configuration parameters."""
    params = KVCacheParams(
        dtype=DType.float32,
        n_kv_heads=16,
        head_dim=64,
        num_layers=1,
        enable_prefix_caching=True,
        kv_connector=KVConnectorType.local,
        host_kvcache_swap_space_gb=10.5,
        page_size=32,
        is_mla=True,
        num_q_heads=128,
        devices=[DeviceRef.GPU(i) for i in range(8)],
        data_parallel_degree=8,
    )

    copied = params.copy_as_dp_1()

    # Verify adjusted parameters
    assert copied.data_parallel_degree == 1
    assert copied.n_devices == 1
    assert copied.n_kv_heads_per_device == 1

    # Verify all other parameters are preserved
    assert copied.dtype == DType.float32
    assert copied.n_kv_heads == 16
    assert copied.head_dim == 64
    assert copied.enable_prefix_caching is True
    assert copied.kv_connector == KVConnectorType.local
    assert copied.host_kvcache_swap_space_gb == 10.5
    assert copied.page_size == 32
    assert copied.is_mla is True


def test_copy_as_dp_1_runs_post_init_validation() -> None:
    """Test that copy_as_dp_1 runs __post_init__ validation."""
    # Create a DP config that will become invalid TP when copied
    # 8 devices with DP=8, becomes 1 device with DP=1
    # But with 7 heads (not divisible by 1), it should still be valid
    params = KVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=7,
        head_dim=128,
        num_layers=1,
        devices=[DeviceRef.GPU(i) for i in range(8)],
        data_parallel_degree=8,
        page_size=16,
    )

    # This should work because 1 device with 7 heads is valid
    copied = params.copy_as_dp_1()
    assert copied.n_devices == 1
    assert copied.n_kv_heads_per_device == 7


def test_copy_as_dp_1_from_dp_1() -> None:
    """Test copy_as_dp_1 when data_parallel_degree is already 1."""
    params = KVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=8,
        head_dim=128,
        num_layers=1,
        devices=[DeviceRef.GPU(i) for i in range(2)],
        data_parallel_degree=1,
        page_size=16,
    )

    copied = params.copy_as_dp_1()

    # Everything should remain the same
    assert copied.data_parallel_degree == 1
    assert copied.n_devices == 2
    assert copied.n_kv_heads_per_device == 4


def test_copy_as_dp_1_non_divisible_devices_fails() -> None:
    """Test that copy_as_dp_1 fails when n_devices is not divisible by DP degree."""
    params = KVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=8,
        head_dim=128,
        num_layers=1,
        devices=[DeviceRef.GPU(i) for i in range(5)],
        data_parallel_degree=5,
        page_size=16,
    )

    # Manually set n_devices to a non-divisible value to test the check
    # (This is a bit contrived since __post_init__ would normally prevent this)
    params.devices = [DeviceRef.GPU(i) for i in range(7)]

    with pytest.raises(
        ValueError,
        match=r"Number of devices \(7\) must be evenly divisible by data parallel degree \(5\)",
    ):
        params.copy_as_dp_1()


def test_copy_as_dp_1_does_not_modify_original() -> None:
    """Test that copy_as_dp_1 does not modify the original instance."""
    params = KVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=8,
        head_dim=128,
        num_layers=1,
        devices=[DeviceRef.GPU(i) for i in range(4)],
        data_parallel_degree=4,
        page_size=16,
    )

    original_devices = params.devices
    original_dp = params.data_parallel_degree
    original_heads_per_device = params.n_kv_heads_per_device

    copied = params.copy_as_dp_1()

    # Verify original is unchanged
    assert params.devices == original_devices
    assert params.data_parallel_degree == original_dp
    assert params.n_kv_heads_per_device == original_heads_per_device

    # Verify n_devices and data_parallel_degree changed in the copy
    assert copied.devices == [DeviceRef.GPU()]
    assert copied.data_parallel_degree == 1
    assert copied.n_kv_heads_per_device == 8


def test_mla_bypasses_divisibility_check() -> None:
    """Test MLA mode bypasses tensor parallel head divisibility check."""
    # This would fail for non-MLA due to 1 head not being divisible by 4 devices
    params = KVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=1,
        head_dim=576,
        num_layers=1,
        devices=[DeviceRef.GPU(i) for i in range(4)],
        data_parallel_degree=1,
        page_size=128,
        is_mla=True,
        num_q_heads=128,
    )
    assert params.n_kv_heads == 1
    assert params.n_kv_heads_per_device == 1


def test_mla_with_data_parallel_compatible() -> None:
    """Test MLA mode with data parallelism."""
    params = KVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=1,
        head_dim=576,
        num_layers=1,
        devices=[DeviceRef.GPU(i) for i in range(4)],
        data_parallel_degree=4,
        page_size=128,
        is_mla=True,
        num_q_heads=128,
    )
    # In DP mode, all heads are on each device
    assert params.n_kv_heads_per_device == 1


def test_kv_cache_quantization_config() -> None:
    kv_cache_quant_config = KVCacheQuantizationConfig(
        quantization_granularity=64
    )
    dp: int = 2
    tp: int = 1
    params = KVCacheParams(
        dtype=DType.float8_e4m3fn,
        n_kv_heads=8,
        head_dim=128,
        num_layers=1,
        page_size=128,
        data_parallel_degree=dp,
        devices=[DeviceRef.GPU(i) for i in range(tp * dp)],
        kvcache_quant_config=kv_cache_quant_config,
    )
    assert params.kvcache_quant_config is not None
    assert params.kvcache_quant_config.quantization_granularity == 64
    assert params.quantized_kv_cache


# ==================== AttnKey Tests ====================


def test_attn_key_is_attn_key_subclass() -> None:
    assert isinstance(
        MHAAttnKey(batch_size=1, max_prompt_length=1, num_partitions=1),
        AttnKey,
    )
    assert isinstance(
        MLAAttnKey(batch_size=1, max_prompt_length=1, num_partitions=1),
        AttnKey,
    )


def test_mha_attn_key_pack_into_buffer() -> None:
    """MHA packs a 4-int CPU buffer ending in max_cache_valid_length."""
    key = MHAAttnKey(batch_size=2, max_prompt_length=1, num_partitions=5)
    buffer = key.pack_into_buffer(CPU(), max_cache_valid_length=123)
    np.testing.assert_array_equal(
        buffer.to_numpy(), np.array([2, 1, 5, 123], dtype=np.int64)
    )


def test_mla_attn_key_pack_into_buffer_ignores_cache_length() -> None:
    """MLA packs a 3-int buffer; max_cache_valid_length is not included."""
    key = MLAAttnKey(batch_size=2, max_prompt_length=1, num_partitions=5)
    buffer = key.pack_into_buffer(CPU(), max_cache_valid_length=123)
    np.testing.assert_array_equal(
        buffer.to_numpy(), np.array([2, 1, 5], dtype=np.int64)
    )


def test_batch_characteristics_is_hashable() -> None:
    """BatchCharacteristics is a frozen, hashable dataclass (dict key)."""
    bc1 = BatchCharacteristics(
        batch_size=1, max_prompt_length=1, max_cache_valid_length=64
    )
    bc2 = BatchCharacteristics(
        batch_size=1, max_prompt_length=1, max_cache_valid_length=64
    )
    assert bc1 == bc2
    assert hash(bc1) == hash(bc2)
    assert {bc1: "x"}[bc2] == "x"


# ==================== Probe-length Tests ====================


def test_graph_capture_probe_cache_lengths_mha() -> None:
    """MHA probes at 256-token granularity."""
    params = KVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=8,
        head_dim=128,
        num_layers=1,
        devices=[DeviceRef.CPU()],
        page_size=16,
    )
    assert params.graph_capture_probe_cache_lengths(1000, 1) == [
        1,
        256,
        512,
        768,
        1000,
    ]


def test_graph_capture_probe_cache_lengths_mla() -> None:
    """MLA probes at 64-token granularity, with extra probes when q > 1."""
    params = KVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=1,
        head_dim=576,
        num_layers=1,
        devices=[DeviceRef.CPU()],
        page_size=128,
        is_mla=True,
        num_q_heads=128,
    )
    base = params.graph_capture_probe_cache_lengths(256, 1)
    assert base == [1, 64, 128, 192, 256]

    # Speculative decoding (q > 1) adds the granularity-1 offset probes.
    spec = params.graph_capture_probe_cache_lengths(256, 2)
    assert set(base).issubset(set(spec))
    assert 63 in spec and 127 in spec and 191 in spec


def test_graph_capture_probe_cache_lengths_filters_min_cache() -> None:
    """Probe lengths below the decode footprint (1 + 2*num_draft_tokens) drop.

    A speculative decode step always caches at least the verified draft tokens
    plus the newly written draft tokens, so a captured graph for a smaller cache
    length is never replayed (and can't be prepared with the dummy warmup
    batch). The probe set must exclude those lengths.
    """
    params = KVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=8,
        head_dim=128,
        num_layers=1,
        devices=[DeviceRef.CPU()],
        page_size=16,
        num_draft_tokens=3,
    )
    # min_cache_length = 1 + 2 * 3 = 7; the length-1 probe is dropped.
    probes = params.graph_capture_probe_cache_lengths(1000, 1)
    assert min(probes) >= 7
    assert 1 not in probes
    assert 256 in probes  # larger probes are unaffected


def test_graph_capture_probe_cache_lengths_no_draft_keeps_length_one() -> None:
    """Without draft tokens the footprint is 1, so no probes are filtered."""
    params = KVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=8,
        head_dim=128,
        num_layers=1,
        devices=[DeviceRef.CPU()],
        page_size=16,
    )
    assert 1 in params.graph_capture_probe_cache_lengths(1000, 1)
