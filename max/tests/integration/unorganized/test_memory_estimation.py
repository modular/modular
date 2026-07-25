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

import logging
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
from max.driver import CPU, DeviceSpec, load_devices
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.comm import Signals
from max.nn.kv_cache import MHAKVCacheParams, MultiKVCacheParams
from max.nn.kv_cache.cache_params import KVConnectorType
from max.pipelines.kv_cache.memory_planner import PagedMemoryPlanner
from max.pipelines.lib import MemoryEstimator
from max.pipelines.lib.interfaces import (
    ArchConfigWithKVCache,
)
from max.pipelines.lib.memory_estimation import (
    _kv_params_per_layer_depth,
    _max_per_layer_buffer_count,
)
from test_common.mocks import DummyPipelineConfig
from test_common.pipeline_model_dummy import (
    DUMMY_LLAMA_ARCH,
    DummyLlamaPipelineModel,
)


def _mha_params(num_layers: int, per_layer_buffers: bool) -> MHAKVCacheParams:
    return MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=8,
        head_dim=128,
        num_layers=num_layers,
        devices=[DeviceRef.GPU()],
        page_size=128,
        per_layer_buffers=per_layer_buffers,
    )


def test_per_layer_depth_single_buffer_pool_is_one() -> None:
    """A normal single-buffer pool contributes a multiplier of 1."""
    assert _kv_params_per_layer_depth(_mha_params(50, False)) == 1


def test_per_layer_depth_reports_num_layers() -> None:
    """A per-layer pool contributes its layer count as the multiplier."""
    assert _kv_params_per_layer_depth(_mha_params(50, True)) == 50


def test_per_layer_depth_tree_takes_largest_flagged_child() -> None:
    """In a tree, only per-layer children count; the largest wins."""
    root = MultiKVCacheParams.from_params(
        {
            "sliding_attention": _mha_params(50, True),
            "full_attention": _mha_params(10, False),
        }
    )
    assert _kv_params_per_layer_depth(root) == 50


def test_per_layer_depth_tree_all_single_buffer_is_one() -> None:
    """A tree with no per-layer child keeps the multiplier at 1."""
    root = MultiKVCacheParams.from_params(
        {"a": _mha_params(50, False), "b": _mha_params(10, False)}
    )
    assert _kv_params_per_layer_depth(root) == 1


def test_max_per_layer_buffer_count_non_kv_arch_is_one() -> None:
    """A non-KV arch config leaves the allocation cap unchanged."""
    assert _max_per_layer_buffer_count(object()) == 1  # type: ignore[arg-type]


def test_max_per_layer_buffer_count_reads_arch_kv_params() -> None:
    """The multiplier is read from the arch config's KV params."""
    arch = MagicMock(spec=ArchConfigWithKVCache)
    arch.get_kv_params.return_value = _mha_params(50, True)
    assert _max_per_layer_buffer_count(arch) == 50


def test_memory_estimation__raise_oom_error_weights_size_exceeds_available_memory() -> (
    None
):
    with (
        patch.object(
            DummyLlamaPipelineModel,
            "calculate_max_seq_len",
            return_value=100000,
        ),
        patch(
            "max.driver.Device.stats", new_callable=PropertyMock
        ) as device_mock,
    ):
        device_mock.return_value = {"free_memory": 5 * 1024 * 1024}
        with pytest.raises(
            RuntimeError, match="Model size exceeds available memory"
        ):
            mock_config = DummyPipelineConfig(
                model_path="modularai/Llama-3.1-8B-Instruct-GGUF",
                max_batch_size=None,
                max_length=None,
                device_specs=[],
                quantization_encoding=DUMMY_LLAMA_ARCH.default_encoding,
            )

            devices = load_devices(mock_config.model.device_specs)
            arch_config = DUMMY_LLAMA_ARCH.config.initialize(mock_config)
            MemoryEstimator.estimate_memory_footprint(
                mock_config,
                mock_config.model,
                arch_config,
                devices,
                50 * 1024 * 1024,
                0,
            )


def test_memory_estimation__infer_optimal_batch_size() -> None:
    # Max batch size on CPU is always 1.
    inferred_batch_size = MemoryEstimator._infer_optimal_batch_size(
        arch_config=MagicMock(spec=ArchConfigWithKVCache),
        devices=[CPU()],
    )
    assert inferred_batch_size == 1


@pytest.mark.skip("TODO: AITLIB-238")
def test_memory_estimation__raise_oom_error_all_defaults_no_valid_solution() -> (
    None
):
    with (
        patch(
            "max.driver.Device.stats", new_callable=PropertyMock
        ) as device_mock,
    ):
        device_mock.return_value = {"free_memory": 30641 * 1024 * 1024}
        with pytest.raises(
            RuntimeError,
        ):
            mock_config = DummyPipelineConfig(
                model_path="modularai/Llama-3.1-8B-Instruct-GGUF",
                max_batch_size=None,
                max_length=None,
                device_specs=[],
                quantization_encoding=DUMMY_LLAMA_ARCH.default_encoding,
            )
            devices = load_devices(mock_config.model.device_specs)
            arch_config = DUMMY_LLAMA_ARCH.config.initialize(mock_config)
            MemoryEstimator.estimate_memory_footprint(
                mock_config,
                mock_config.model,
                arch_config,
                devices,
                30000 * 1024 * 1024,
                0,
            )


@pytest.mark.skip("TODO: AITLIB-293, Use accurate mocked values")
def test_memory_estimation__raise_oom_error_all_defaults(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with (
        patch(
            "max.driver.Device.stats", new_callable=PropertyMock
        ) as device_mock,
    ):
        device_mock.return_value = {"free_memory": 40000 * 1024 * 1024}
        with caplog.at_level(logging.WARNING):
            mock_config = DummyPipelineConfig(
                model_path="modularai/Llama-3.1-8B-Instruct-GGUF",
                max_batch_size=None,
                max_length=None,
                device_specs=[],
                quantization_encoding=DUMMY_LLAMA_ARCH.default_encoding,
            )
            devices = load_devices(mock_config.model.device_specs)
            arch_config = DUMMY_LLAMA_ARCH.config.initialize(mock_config)
            MemoryEstimator.estimate_memory_footprint(
                mock_config,
                mock_config.model,
                arch_config,
                devices,
                35000 * 1024 * 1024,
                0,
            )

        assert "Truncated model's default max_length from" in caplog.text


@pytest.mark.skip("TODO: AITLIB-293, Use accurate mocked values")
def test_memory_estimation__raise_oom_error_max_length_set() -> None:
    with (
        patch(
            "max.driver.Device.stats", new_callable=PropertyMock
        ) as device_mock,
    ):
        device_mock.return_value = {"free_memory": 40000 * 1024 * 1024}
        with pytest.raises(
            RuntimeError,
            match=r"Try reducing --max-length to \d+ .*supports batch size of",
        ):
            mock_config = DummyPipelineConfig(
                model_path="modularai/Llama-3.1-8B-Instruct-GGUF",
                max_batch_size=None,
                max_length=100000,
                device_specs=[],
                quantization_encoding=DUMMY_LLAMA_ARCH.default_encoding,
            )
            devices = load_devices(mock_config.model.device_specs)
            arch_config = DUMMY_LLAMA_ARCH.config.initialize(mock_config)
            MemoryEstimator.estimate_memory_footprint(
                mock_config,
                mock_config.model,
                arch_config,
                devices,
                35000 * 1024 * 1024,
                0,
            )


@pytest.mark.skip("TODO: AITLIB-293, Use accurate mocked values")
def test_memory_estimation__raise_oom_error_max_batch_size_set() -> None:
    with (
        patch.object(
            DummyLlamaPipelineModel, "calculate_max_seq_len", return_value=4096
        ),
        patch(
            "max.driver.Device.stats", new_callable=PropertyMock
        ) as device_mock,
    ):
        device_mock.return_value = {"free_memory": 40000 * 1024 * 1024}
        with pytest.raises(RuntimeError, match="reducing --max-batch-size to"):
            mock_config = DummyPipelineConfig(
                model_path="modularai/Llama-3.1-8B-Instruct-GGUF",
                max_batch_size=100000,
                max_length=None,
                device_specs=[],
                quantization_encoding=DUMMY_LLAMA_ARCH.default_encoding,
            )
            devices = load_devices(mock_config.model.device_specs)
            arch_config = DUMMY_LLAMA_ARCH.config.initialize(mock_config)
            MemoryEstimator.estimate_memory_footprint(
                mock_config,
                mock_config.model,
                arch_config,
                devices,
                40000 * 1024 * 1024,
                0,
            )


@pytest.mark.skip("TODO: AITLIB-293, Use accurate mocked values")
def test_memory_estimation__raise_oom_error_max_batch_size_set_and_max_length_set() -> (
    None
):
    with (
        patch(
            "max.driver.Device.stats", new_callable=PropertyMock
        ) as device_mock,
    ):
        device_mock.return_value = {"free_memory": 40000 * 1024 * 1024}
        with pytest.raises(RuntimeError, match="reducing --max-batch-size to"):
            mock_config = DummyPipelineConfig(
                model_path="modularai/Llama-3.1-8B-Instruct-GGUF",
                max_batch_size=100000,
                max_length=4096,
                device_specs=[],
                quantization_encoding=DUMMY_LLAMA_ARCH.default_encoding,
            )
            devices = load_devices(mock_config.model.device_specs)
            arch_config = DUMMY_LLAMA_ARCH.config.initialize(mock_config)
            MemoryEstimator.estimate_memory_footprint(
                mock_config,
                mock_config.model,
                arch_config,
                devices,
                40000 * 1024 * 1024,
                0,
            )


@pytest.mark.parametrize(
    "device_specs,kv_connector,expected_count_per_gpu",
    [
        # Single-device: no signal buffers in the default path.
        ([DeviceSpec.cpu()], KVConnectorType.null, 0),
        ([DeviceSpec.accelerator(id=0)], KVConnectorType.null, 0),
        # Multi-GPU baseline: one set per device for the main model.
        (
            [DeviceSpec.accelerator(id=i) for i in range(2)],
            KVConnectorType.null,
            1,
        ),
        (
            [DeviceSpec.accelerator(id=i) for i in range(4)],
            KVConnectorType.null,
            1,
        ),
        (
            [DeviceSpec.accelerator(id=i) for i in range(8)],
            KVConnectorType.null,
            1,
        ),
        # KV-offload adds BlockOffloadEngine's set.
        (
            [DeviceSpec.accelerator(id=i) for i in range(2)],
            KVConnectorType.tiered,
            2,
        ),
        (
            [DeviceSpec.accelerator(id=i) for i in range(4)],
            KVConnectorType.local,
            2,
        ),
        (
            [DeviceSpec.accelerator(id=i) for i in range(8)],
            KVConnectorType.tiered,
            2,
        ),
        # dkv connector doesn't allocate BlockOffloadEngine.
        (
            [DeviceSpec.accelerator(id=i) for i in range(2)],
            KVConnectorType.dkv,
            1,
        ),
    ],
)
def test_estimate_signal_buffer_memory__default(
    device_specs: list[DeviceSpec],
    kv_connector: KVConnectorType,
    expected_count_per_gpu: int,
) -> None:
    """``PipelineConfig.estimate_signal_buffer_memory`` returns
    ``NUM_BYTES * count_per_gpu * ngpus`` for the in-scope allocation sites."""
    cfg = DummyPipelineConfig(
        model_path="dummy",
        quantization_encoding=DUMMY_LLAMA_ARCH.default_encoding,
        max_batch_size=1,
        max_length=1024,
        device_specs=device_specs,
    )
    cfg.model.kv_cache.kv_connector = kv_connector

    expected = Signals.NUM_BYTES * expected_count_per_gpu * len(device_specs)
    assert cfg.estimate_signal_buffer_memory() == expected


@pytest.mark.parametrize(
    "ngpus,kv_connector,expected_count_per_gpu",
    [
        # Single-GPU: mixin allocates one set even though the default would not.
        (1, KVConnectorType.null, 1),
        # Multi-GPU: mixin matches the default.
        (2, KVConnectorType.null, 1),
        (4, KVConnectorType.tiered, 2),
        (8, KVConnectorType.local, 2),
    ],
)
def test_estimate_signal_buffer_memory__always_signal_buffers_mixin(
    ngpus: int,
    kv_connector: KVConnectorType,
    expected_count_per_gpu: int,
) -> None:
    """Planners with ``always_signal_buffers=True`` allocate one set even at
    single-GPU and match the default for multi-GPU."""
    device_specs = [DeviceSpec.accelerator(id=i) for i in range(ngpus)]
    cfg = DummyPipelineConfig(
        model_path="dummy",
        quantization_encoding=DUMMY_LLAMA_ARCH.default_encoding,
        max_batch_size=1,
        max_length=1024,
        device_specs=device_specs,
    )
    cfg.model.kv_cache.kv_connector = kv_connector

    arch_config = DUMMY_LLAMA_ARCH.config.initialize(cfg)
    planner_cls = PagedMemoryPlanner.with_activation_reservation(
        0, always_signal_buffers=True
    )
    planner = planner_cls(arch_config)
    got = planner.estimate_signal_buffer_memory(cfg)
    expected = Signals.NUM_BYTES * expected_count_per_gpu * max(ngpus, 1)
    assert got == expected


@pytest.mark.parametrize(
    "replicates_kv_across_tp,expected_count_per_gpu",
    [
        # BlockOffloadEngine only allocates signal buffers when the KV cache
        # is replicated across TP (is_mla AND dp==1 AND n_devices>1).
        (True, 2),  # main model + BCE
        (False, 1),  # main model only, BCE skips signal-buffer setup
    ],
)
def test_estimate_signal_buffer_memory__bce_gated_by_kv_params(
    replicates_kv_across_tp: bool,
    expected_count_per_gpu: int,
) -> None:
    """With an ``arch_config`` exposing :class:`KVCacheParamInterface`,
    the BCE term is gated on ``replicates_kv_across_tp``."""
    device_specs = [DeviceSpec.accelerator(id=i) for i in range(4)]
    cfg = DummyPipelineConfig(
        model_path="dummy",
        quantization_encoding=DUMMY_LLAMA_ARCH.default_encoding,
        max_batch_size=1,
        max_length=1024,
        device_specs=device_specs,
    )
    cfg.model.kv_cache.kv_connector = KVConnectorType.tiered

    arch_config = MagicMock(spec=ArchConfigWithKVCache)
    arch_config.get_kv_params.return_value.replicates_kv_across_tp = (
        replicates_kv_across_tp
    )

    got = cfg.estimate_signal_buffer_memory(arch_config)
    expected = Signals.NUM_BYTES * expected_count_per_gpu * len(device_specs)
    assert got == expected
