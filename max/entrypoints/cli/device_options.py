# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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


"""Custom Click Options used in pipelines"""

from __future__ import annotations

import logging
from typing import Any

import click
from max.driver import DeviceSpec, scan_available_devices

logger = logging.getLogger("max.pipelines")


class DevicesOptionType(click.ParamType):
    name = "devices"

    @staticmethod
    def _get_requested_gpu_ids(devices: str | list[int]) -> list[int]:
        """Helper function to get requested GPU IDs from devices input.

        Args:
            devices: The devices input, either "gpu" or a list of GPU IDs

        Returns:
            List of requested GPU IDs
        """
        if devices == "gpu" or devices == "default":
            return [0]
        elif isinstance(devices, list):
            return devices
        return []

    @staticmethod
    def _validate_gpu_ids(
        gpu_ids: list[int], available_gpu_ids: list[int]
    ) -> None:
        """Helper function to validate requested GPU IDs against available ones.

        Args:
            gpu_ids: List of requested GPU IDs
            available_gpu_ids: List of available GPU IDs

        Raises:
            ValueError: If a requested GPU ID is not available
        """
        for gpu_id in gpu_ids:
            if gpu_id not in available_gpu_ids:
                if len(available_gpu_ids) == 0:
                    msg = (
                        f"GPU {gpu_id} requested but no GPUs are available. "
                        f"Use valid device IDs or '--devices=cpu'."
                    )
                else:
                    msg = (
                        f"GPU {gpu_id} requested but only GPU IDs {available_gpu_ids} are "
                        f"available. Use valid device IDs or '--devices=cpu'."
                    )
                raise ValueError(msg)

    @staticmethod
    def device_specs(devices: str | list[int]) -> list[DeviceSpec]:
        """Converts parsed devices input into validated :obj:`DeviceSpec` objects.

        Args:
            devices: The value provided by the --devices option.
                Valid arguments:
                - "cpu"   → use the CPU,
                - "gpu"   → default to GPU 0, or,
                - a list of ints (GPU IDs).

        Raises:
            ValueError: If a requested GPU ID is invalid.

        Returns:
            A list of DeviceSpec objects.
        """
        available_devices = scan_available_devices()
        available_gpu_ids = [
            d.id for d in available_devices if d.device_type == "gpu"
        ]
        if devices == "cpu":
            return [DeviceSpec.cpu()]
        elif devices == "default" and len(available_gpu_ids) == 0:
            logger.info("No GPUs available, falling back to CPU")
            return [DeviceSpec.cpu()]

        requested_gpu_ids = DevicesOptionType._get_requested_gpu_ids(devices)

        DevicesOptionType._validate_gpu_ids(
            requested_gpu_ids, available_gpu_ids
        )

        return [DeviceSpec.accelerator(id=id) for id in requested_gpu_ids]

    @staticmethod
    def parse_from_str(value: str) -> str | list[int]:
        """Parse a device string into either a string or list of ints.

        Args:
            value: The value provided as a string (e.g., "cpu", "gpu", "gpu:0,1,2")

        Returns:
            Either "cpu", "gpu", "default", or a non-empty list of GPU IDs as integers.

        Raises:
            ValueError: If the format is invalid
        """
        if value.lower() in {"cpu", "gpu", "default"}:
            return value.lower()

        # By this point, we should only be left with a list of GPU IDs in a
        # gpu:<id1>,<id2> format.
        if not value.startswith("gpu:"):
            raise ValueError(
                f"Expected 'gpu:<id1>,<id2>' format, got '{value}'"
            )
        # Remove the "gpu:" prefix and split the string by commas to get a list of GPU IDs.
        try:
            gpu_ids = value.removeprefix("gpu:").split(",")
            return [int(part) for part in gpu_ids]
        except ValueError:
            raise ValueError(
                f"'{value}' is not a valid device list. Use format 'cpu', 'gpu', or 'gpu:0,1'."
            )

    def convert(
        self,
        value: Any,
        param: click.Parameter | None = None,
        ctx: click.Context | None = None,
    ) -> str | list[int]:
        try:
            return self.parse_from_str(value)
        except ValueError as e:
            self.fail(str(e), param, ctx)
