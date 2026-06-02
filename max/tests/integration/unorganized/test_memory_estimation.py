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
from max.nn.comm import Signals
from max.nn.kv_cache.cache_params import KVConnectorType
from max.pipelines.lib import MemoryEstimator
from max.pipelines.lib.interfaces import (
    AlwaysSignalBuffersMixin,
    ArchConfigWithKVCache,
)
from test_common.mocks import DummyPipelineConfig
from test_common.pipeline_model_dummy import (
    DUMMY_LLAMA_ARCH,
    DummyLlamaPipelineModel,
)


def test_memory_estimation__raise_oom_error_weights_size_exceeds_available_memory() -> (
    None
):
    with (
        patch.object(
            DummyLlamaPipelineModel,
            "calculate_max_seq_len",
            return_value=100000,
        ),
        patch.object(
            DummyLlamaPipelineModel,
            "estimate_weights_size",
            return_value=50 * 1024 * 1024,
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
                DummyLlamaPipelineModel.estimate_weights_size(mock_config),
                DummyLlamaPipelineModel.estimate_activation_memory(
                    mock_config, mock_config.model.huggingface_config
                ),
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
        patch.object(
            DummyLlamaPipelineModel,
            "estimate_weights_size",
            return_value=30000 * 1024 * 1024,
        ),
        patch.object(
            DummyLlamaPipelineModel,
            "estimate_activation_memory",
            return_value=0,
        ),
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
                DummyLlamaPipelineModel.estimate_weights_size(mock_config),
                DummyLlamaPipelineModel.estimate_activation_memory(
                    mock_config, mock_config.model.huggingface_config
                ),
            )


@pytest.mark.skip("TODO: AITLIB-293, Use accurate mocked values")
def test_memory_estimation__raise_oom_error_all_defaults(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with (
        patch.object(
            DummyLlamaPipelineModel,
            "estimate_weights_size",
            return_value=35000 * 1024 * 1024,
        ),
        patch.object(
            DummyLlamaPipelineModel,
            "estimate_activation_memory",
            return_value=0,
        ),
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
                DummyLlamaPipelineModel.estimate_weights_size(mock_config),
                DummyLlamaPipelineModel.estimate_activation_memory(
                    mock_config, mock_config.model.huggingface_config
                ),
            )

        assert "Truncated model's default max_length from" in caplog.text


@pytest.mark.skip("TODO: AITLIB-293, Use accurate mocked values")
def test_memory_estimation__raise_oom_error_max_length_set() -> None:
    with (
        patch.object(
            DummyLlamaPipelineModel,
            "estimate_weights_size",
            return_value=35000 * 1024 * 1024,
        ),
        patch.object(
            DummyLlamaPipelineModel,
            "estimate_activation_memory",
            return_value=0,
        ),
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
                DummyLlamaPipelineModel.estimate_weights_size(mock_config),
                DummyLlamaPipelineModel.estimate_activation_memory(
                    mock_config, mock_config.model.huggingface_config
                ),
            )


@pytest.mark.skip("TODO: AITLIB-293, Use accurate mocked values")
def test_memory_estimation__raise_oom_error_max_batch_size_set() -> None:
    with (
        patch.object(
            DummyLlamaPipelineModel, "calculate_max_seq_len", return_value=4096
        ),
        patch.object(
            DummyLlamaPipelineModel,
            "estimate_weights_size",
            return_value=40000 * 1024 * 1024,
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
                DummyLlamaPipelineModel.estimate_weights_size(mock_config),
                DummyLlamaPipelineModel.estimate_activation_memory(
                    mock_config, mock_config.model.huggingface_config
                ),
            )


@pytest.mark.skip("TODO: AITLIB-293, Use accurate mocked values")
def test_memory_estimation__raise_oom_error_max_batch_size_set_and_max_length_set() -> (
    None
):
    with (
        patch.object(
            DummyLlamaPipelineModel,
            "estimate_weights_size",
            return_value=40000 * 1024 * 1024,
        ),
        patch.object(
            DummyLlamaPipelineModel,
            "estimate_activation_memory",
            return_value=0,
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
                DummyLlamaPipelineModel.estimate_weights_size(mock_config),
                DummyLlamaPipelineModel.estimate_activation_memory(
                    mock_config, mock_config.model.huggingface_config
                ),
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
    """``AlwaysSignalBuffersMixin`` allocates one set even at single-GPU,
    and matches the default for multi-GPU."""
    device_specs = [DeviceSpec.accelerator(id=i) for i in range(ngpus)]
    cfg = DummyPipelineConfig(
        model_path="dummy",
        quantization_encoding=DUMMY_LLAMA_ARCH.default_encoding,
        max_batch_size=1,
        max_length=1024,
        device_specs=device_specs,
    )
    cfg.model.kv_cache.kv_connector = kv_connector

    got = AlwaysSignalBuffersMixin.estimate_signal_buffer_memory(cfg)
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
