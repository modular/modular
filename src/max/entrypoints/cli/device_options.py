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

from typing import Any

import click
from max.driver import DeviceSpec, accelerator_count


class DevicesOptionType(click.ParamType):
    name = "devices"

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
        num_available_gpus = accelerator_count()
        if devices == "cpu" or num_available_gpus == 0:
            return [DeviceSpec.cpu()]

        requested_ids: list[int] = []
        if devices == "gpu":
            requested_ids = [0]
        elif isinstance(devices, list):
            requested_ids = devices

        if not requested_ids:
            # Return device 0 when no specific IDs are requested.
            return [DeviceSpec.accelerator(id=0)]

        # Validate requested GPU IDs.
        for gpu_id in requested_ids:
            if gpu_id >= num_available_gpus:
                msg = (
                    f"GPU {gpu_id} requested but only {num_available_gpus} "
                    "available. Use valid device IDs or '--devices=cpu'."
                )
                raise ValueError(msg)

        return [DeviceSpec.accelerator(id=id) for id in requested_ids]

    @staticmethod
    def parse_from_str(value: str) -> str | list[int]:
        """Parse a device string into either a string or list of ints.

        Args:
            value: The value provided as a string (e.g., "cpu", "gpu", "gpu:0,1,2")

        Returns:
            Either "cpu", "gpu", or a list of GPU IDs as integers

        Raises:
            ValueError: If the format is invalid
        """
        if not value:
            return []

        if value.lower() in {"cpu", "gpu"}:
            return value.lower()

        try:
            # Support both "gpu:0,1" and old "0,1" formats.
            return [int(part) for part in value.replace("gpu:", "").split(",")]
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
