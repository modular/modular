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
"""Tests for device specs parsing helpers."""

from __future__ import annotations

import pytest
from max.driver import DeviceSpec
from max.pipelines.lib import device_specs
from max.pipelines.lib.device_specs import (
    DeviceHandle,
    coerce_device_specs_input,
    get_requested_gpu_ids,
    normalize_device_specs_input,
    validate_gpu_ids,
)
from max.pipelines.lib.device_specs import (
    device_specs_from_normalized_device_handle as ds_from_ndh,
)
from pytest_mock import MockerFixture


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("cpu", "cpu"),
        ("CPU", "cpu"),
        ("gpu", "gpu"),
        ("GPU", "gpu"),
        ("default", "default"),
        ("DeFaUlT", "default"),
    ],
)
def test_normalize_device_specs_input_named_values(
    value: str, expected: str
) -> None:
    assert normalize_device_specs_input(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("gpu:0", [0]),
        ("gpu:0,1,2", [0, 1, 2]),
        ("gpu:5,7", [5, 7]),
    ],
)
def test_normalize_device_specs_input_gpu_list(
    value: str, expected: list[int]
) -> None:
    assert normalize_device_specs_input(value) == expected


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("tpu", "Expected 'gpu:<id1>,<id2>' format, got 'tpu'"),
        (
            "gpu:a,1",
            "'gpu:a,1' is not a valid device list. Use format 'cpu', 'gpu', or 'gpu:0,1'.",
        ),
        (
            "gpu:",
            "'gpu:' is not a valid device list. Use format 'cpu', 'gpu', or 'gpu:0,1'.",
        ),
    ],
)
def test_normalize_device_specs_input_invalid_values(
    value: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_device_specs_input(value)


@pytest.mark.parametrize(
    ("devices", "expected"),
    [
        ("gpu", [0]),
        ("default", [0]),
        ([2, 3], [2, 3]),
        ("cpu", []),
    ],
)
def test_get_requested_gpu_ids(
    devices: DeviceHandle, expected: list[int]
) -> None:
    assert get_requested_gpu_ids(devices=devices) == expected


def test_validate_gpu_ids_allows_no_available_gpus() -> None:
    with pytest.raises(
        ValueError,
        match=r"GPU id 0 requested but no GPUs are available. Use valid device IDs or '--devices=cpu'.",
    ):
        validate_gpu_ids(gpu_ids=[0], available_gpu_ids=[])


def test_validate_gpu_ids_raises_on_invalid_gpu() -> None:
    with pytest.raises(
        ValueError,
        match=(
            r"GPU id 2 requested but only GPU IDs \[0, 1\] are available\. "
            r"Use valid device IDs or '--devices=cpu'\."
        ),
    ):
        validate_gpu_ids(gpu_ids=[2], available_gpu_ids=[0, 1])


def test_device_specs_from_normalized_device_handle_cpu(
    mocker: MockerFixture,
) -> None:
    mocker.patch.object(
        device_specs,
        "scan_available_devices",
        autospec=True,
        return_value=[DeviceSpec.accelerator(0)],
    )
    assert ds_from_ndh("cpu") == [DeviceSpec.cpu()]


def test_device_specs_from_normalized_device_handle_default_no_gpus(
    mocker: MockerFixture,
) -> None:
    mocker.patch.object(
        device_specs,
        "scan_available_devices",
        autospec=True,
        return_value=[],
    )
    assert ds_from_ndh("default") == [DeviceSpec.cpu()]


def test_device_specs_from_normalized_device_handle_gpu_list(
    mocker: MockerFixture,
) -> None:
    mocker.patch.object(
        device_specs,
        "scan_available_devices",
        autospec=True,
        return_value=[DeviceSpec.accelerator(0), DeviceSpec.accelerator(2)],
    )
    specs = ds_from_ndh([0, 2])
    assert specs == [
        DeviceSpec.accelerator(0),
        DeviceSpec.accelerator(2),
    ]


def test_coerce_device_specs_input_parses_strings(
    mocker: MockerFixture,
) -> None:
    mocker.patch.object(
        device_specs,
        "scan_available_devices",
        autospec=True,
        return_value=[DeviceSpec.cpu()],
    )
    assert coerce_device_specs_input("cpu") == [DeviceSpec.cpu()]
    assert coerce_device_specs_input("gpu") == [DeviceSpec.accelerator(0)]
    assert coerce_device_specs_input("default") == [DeviceSpec.cpu()]
    assert coerce_device_specs_input("gpu:1,2") == [
        DeviceSpec.accelerator(1),
        DeviceSpec.accelerator(2),
    ]


def test_coerce_device_specs_input_passthrough() -> None:
    assert coerce_device_specs_input([1, 3]) == [
        DeviceSpec.accelerator(1),
        DeviceSpec.accelerator(3),
    ]


def test_coerce_device_specs_input_string_list(
    mocker: MockerFixture,
) -> None:
    """CLI parsers like cyclopts may pass device specs as list[str]."""
    mocker.patch.object(
        device_specs,
        "scan_available_devices",
        autospec=True,
        return_value=[DeviceSpec.cpu()],
    )
    # Single-element string list with gpu: prefix
    assert coerce_device_specs_input(["gpu:0,1"]) == [
        DeviceSpec.accelerator(0),
        DeviceSpec.accelerator(1),
    ]
    # Single-element string list with named values
    assert coerce_device_specs_input(["cpu"]) == [DeviceSpec.cpu()]
    assert coerce_device_specs_input(["gpu"]) == [DeviceSpec.accelerator(0)]
    assert coerce_device_specs_input(["default"]) == [DeviceSpec.cpu()]
    # Multi-element numeric string list (e.g. --device-specs 0 1)
    assert coerce_device_specs_input(["0", "1"]) == [
        DeviceSpec.accelerator(0),
        DeviceSpec.accelerator(1),
    ]
    assert coerce_device_specs_input(["3"]) == [DeviceSpec.accelerator(3)]
