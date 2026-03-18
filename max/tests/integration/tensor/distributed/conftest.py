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
"""Shared test helpers for distributed tensor tests."""

from __future__ import annotations

from max.driver import CPU
from max.experimental.distributed import DeviceMesh


def cpu_devices(n: int) -> tuple[CPU, ...]:
    return tuple(CPU() for _ in range(n))


def mesh_1d(n: int, name: str = "tp") -> DeviceMesh:
    return DeviceMesh(
        devices=cpu_devices(n),
        mesh_shape=(n,),
        axis_names=(name,),
    )


def mesh_2d(rows: int, cols: int) -> DeviceMesh:
    return DeviceMesh(
        devices=cpu_devices(rows * cols),
        mesh_shape=(rows, cols),
        axis_names=("dp", "tp"),
    )
