# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import logging
from unittest.mock import PropertyMock, patch

import pytest
from max.driver import load_devices
from max.pipelines.memory_estimation import MEMORY_ESTIMATOR
from test_common.mocks import DummyPipelineConfig
from test_common.pipeline_model_dummy import DUMMY_ARCH, DummyLlamaPipelineModel


def test_memory_estimation__raise_oom_error_weights_size_exceeds_available_memory():
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
                model_path="modularai/llama-3.1",
                max_batch_size=None,
                max_length=None,
                device_specs=[],
                quantization_encoding=DUMMY_ARCH.default_encoding,
            )

            devices = load_devices(mock_config.model_config.device_specs)
            MEMORY_ESTIMATOR.estimate_memory_footprint(
                mock_config,
                DUMMY_ARCH.pipeline_model,
                mock_config.model_config,
                devices,
            )


@pytest.mark.skip("TODO: AITLIB-238")
def test_memory_estimation__raise_oom_error_all_defaults_no_valid_solution():
    with (
        patch.object(
            DummyLlamaPipelineModel, "calculate_max_seq_len", return_value=10000
        ),
        patch.object(
            DummyLlamaPipelineModel,
            "estimate_weights_size",
            return_value=30000 * 1024 * 1024,
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
                model_path="modularai/llama-3.1",
                max_batch_size=None,
                max_length=None,
                device_specs=[],
                quantization_encoding=DUMMY_ARCH.default_encoding,
            )
            devices = load_devices(mock_config.model_config.device_specs)
            MEMORY_ESTIMATOR.estimate_memory_footprint(
                mock_config,
                DUMMY_ARCH.pipeline_model,
                mock_config.model_config,
                devices,
            )


@pytest.mark.skip("TODO: AITLIB-293, Use accurate mocked values")
def test_memory_estimation__raise_oom_error_all_defaults(caplog):
    with (
        patch.object(
            DummyLlamaPipelineModel,
            "calculate_max_seq_len",
            return_value=100000,
        ),
        patch.object(
            DummyLlamaPipelineModel,
            "estimate_weights_size",
            return_value=35000 * 1024 * 1024,
        ),
        patch(
            "max.driver.Device.stats", new_callable=PropertyMock
        ) as device_mock,
    ):
        device_mock.return_value = {"free_memory": 40000 * 1024 * 1024}
        with caplog.at_level(logging.WARNING):
            mock_config = DummyPipelineConfig(
                model_path="modularai/llama-3.1",
                max_batch_size=None,
                max_length=None,
                device_specs=[],
                quantization_encoding=DUMMY_ARCH.default_encoding,
            )
            devices = load_devices(mock_config.model_config.device_specs)
            MEMORY_ESTIMATOR.estimate_memory_footprint(
                mock_config,
                DUMMY_ARCH.pipeline_model,
                mock_config.model_config,
                devices,
            )

        assert "Truncated model's default max_length from" in caplog.text


@pytest.mark.skip("TODO: AITLIB-293, Use accurate mocked values")
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
            mock_config = DummyPipelineConfig(
                model_path="modularai/llama-3.1",
                max_batch_size=None,
                max_length=100000,
                device_specs=[],
                quantization_encoding=DUMMY_ARCH.default_encoding,
            )
            devices = load_devices(mock_config.model_config.device_specs)
            MEMORY_ESTIMATOR.estimate_memory_footprint(
                mock_config,
                DUMMY_ARCH.pipeline_model,
                mock_config.model_config,
                devices,
            )


@pytest.mark.skip("TODO: AITLIB-293, Use accurate mocked values")
def test_memory_estimation__raise_oom_error_max_batch_size_set():
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
                model_path="modularai/llama-3.1",
                max_batch_size=100000,
                max_length=None,
                device_specs=[],
                quantization_encoding=DUMMY_ARCH.default_encoding,
            )
            devices = load_devices(mock_config.model_config.device_specs)
            MEMORY_ESTIMATOR.estimate_memory_footprint(
                mock_config,
                DUMMY_ARCH.pipeline_model,
                mock_config.model_config,
                devices,
            )


@pytest.mark.skip("TODO: AITLIB-293, Use accurate mocked values")
def test_memory_estimation__raise_oom_error_max_batch_size_set_and_max_length_set():
    with (
        patch.object(
            DummyLlamaPipelineModel,
            "calculate_max_seq_len",
            return_value=9999999999999,
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
                model_path="modularai/llama-3.1",
                max_batch_size=100000,
                max_length=4096,
                device_specs=[],
                quantization_encoding=DUMMY_ARCH.default_encoding,
            )
            devices = load_devices(mock_config.model_config.device_specs)
            MEMORY_ESTIMATOR.estimate_memory_footprint(
                mock_config,
                DUMMY_ARCH.pipeline_model,
                mock_config.model_config,
                devices,
            )
