# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import logging
from typing import Optional
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
from max.driver import DeviceSpec, load_devices, scan_available_devices
from max.nn.kv_cache import KVCacheStrategy
from max.pipelines.max_config import KVCacheConfig
from max.pipelines.memory_estimation import MEMORY_ESTIMATOR
from max.pipelines.model_config import MAXModelConfig
from test_common.pipeline_model import DUMMY_ARCH, DummyLlamaPipelineModel


def create_mock_pipeline_config(
    model_path: str,
    max_batch_size: Optional[int],
    max_length: Optional[int],
    device_specs: Optional[list[DeviceSpec]] = None,
) -> MagicMock:
    if device_specs is None:
        device_specs = scan_available_devices()

    mock_config = MagicMock()
    mock_config.model_config = MAXModelConfig(
        model_path=model_path,
        device_specs=device_specs,
        quantization_encoding=DUMMY_ARCH.default_encoding,
        _kv_cache_config=KVCacheConfig(
            cache_strategy=KVCacheStrategy.CONTINUOUS,
        ),
        _huggingface_config=MagicMock(),
    )

    mock_config.max_batch_size = max_batch_size
    mock_config.max_length = max_length
    return mock_config


@pytest.mark.skip("TODO: AITLIB-293")
def test_memory_estimation__raise_oom_error_weights_size_exceeds_available_memory():
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
            mock_config = create_mock_pipeline_config(
                model_path="modularai/llama-3.1",
                max_batch_size=None,
                max_length=None,
            )
            devices = load_devices(mock_config.model_config.device_specs)
            MEMORY_ESTIMATOR.estimate_memory_footprint(
                mock_config, DUMMY_ARCH, devices
            )


@pytest.mark.skip("TODO: AITLIB-238, AITLIB-293")
def test_memory_estimation__raise_oom_error_all_defaults_no_valid_solution():
    with (
        patch.object(
            DummyLlamaPipelineModel, "calculate_max_seq_len", return_value=10000
        ),
        patch(
            "max.driver.Device.stats", new_callable=PropertyMock
        ) as device_mock,
    ):
        device_mock.return_value = {"free_memory": 30641 * 1024 * 1024}
        with pytest.raises(
            RuntimeError,
        ):
            mock_config = create_mock_pipeline_config(
                model_path="modularai/llama-3.1",
                max_batch_size=None,
                max_length=None,
            )
            devices = load_devices(mock_config.model_config.device_specs)
            MEMORY_ESTIMATOR.estimate_memory_footprint(
                mock_config, DUMMY_ARCH, devices
            )


@pytest.mark.skip("TODO: AITLIB-293")
def test_memory_estimation__raise_oom_error_all_defaults(caplog):
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
        device_mock.return_value = {"free_memory": 40000 * 1024 * 1024}
        with caplog.at_level(logging.WARNING):
            mock_config = create_mock_pipeline_config(
                model_path="modularai/llama-3.1",
                max_batch_size=None,
                max_length=None,
            )
            devices = load_devices(mock_config.model_config.device_specs)
            MEMORY_ESTIMATOR.estimate_memory_footprint(
                mock_config, DUMMY_ARCH, devices
            )

        assert "Truncated model's default max_length from" in caplog.text


@pytest.mark.skip("TODO: AITLIB-293")
def test_memory_estimation__raise_oom_error_max_length_set():
    with (
        patch.object(
            DummyLlamaPipelineModel,
            "calculate_max_seq_len",
            return_value=9999999999999,
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
            mock_config = create_mock_pipeline_config(
                model_path="modularai/llama-3.1",
                max_batch_size=None,
                max_length=100000,
            )
            devices = load_devices(mock_config.model_config.device_specs)
            MEMORY_ESTIMATOR.estimate_memory_footprint(
                mock_config, DUMMY_ARCH, devices
            )


@pytest.mark.skip("TODO: AITLIB-293")
def test_memory_estimation__raise_oom_error_max_batch_size_set():
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
            mock_config = create_mock_pipeline_config(
                model_path="modularai/llama-3.1",
                max_batch_size=100000,
                max_length=None,
            )
            devices = load_devices(mock_config.model_config.device_specs)
            MEMORY_ESTIMATOR.estimate_memory_footprint(
                mock_config, DUMMY_ARCH, devices
            )


@pytest.mark.skip("TODO: AITLIB-293")
def test_memory_estimation__raise_oom_error_max_batch_size_set_and_max_length_set():
    with (
        patch.object(
            DummyLlamaPipelineModel,
            "calculate_max_seq_len",
            return_value=9999999999999,
        ),
        patch(
            "max.driver.Device.stats", new_callable=PropertyMock
        ) as device_mock,
    ):
        device_mock.return_value = {"free_memory": 40000 * 1024 * 1024}
        with pytest.raises(RuntimeError, match="reducing --max-batch-size to"):
            mock_config = create_mock_pipeline_config(
                model_path="modularai/llama-3.1",
                max_batch_size=100000,
                max_length=4096,
            )
            devices = load_devices(mock_config.model_config.device_specs)
            MEMORY_ESTIMATOR.estimate_memory_footprint(
                mock_config, DUMMY_ARCH, devices
            )
