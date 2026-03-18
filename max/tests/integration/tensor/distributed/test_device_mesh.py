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
"""Tests for DeviceMesh."""

from __future__ import annotations

import dataclasses

import pytest
from conftest import cpu_devices, mesh_1d, mesh_2d
from max.experimental.distributed import DeviceMesh


class TestDeviceMesh:
    def test_1d_construction(self) -> None:
        mesh = mesh_1d(4)
        assert mesh.ndim == 1
        assert mesh.num_devices == 4
        assert mesh.mesh_shape == (4,)
        assert mesh.axis_names == ("tp",)

    def test_2d_construction(self) -> None:
        mesh = mesh_2d(2, 4)
        assert mesh.ndim == 2
        assert mesh.num_devices == 8
        assert mesh.mesh_shape == (2, 4)
        assert mesh.axis_names == ("dp", "tp")

    def test_device_count_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="expects 8 devices, got 4"):
            DeviceMesh(
                devices=cpu_devices(4),
                mesh_shape=(2, 4),
                axis_names=("dp", "tp"),
            )

    def test_axis_names_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="axis_names length"):
            DeviceMesh(
                devices=cpu_devices(4),
                mesh_shape=(4,),
                axis_names=("dp", "tp"),
            )

    def test_duplicate_axis_names_raises(self) -> None:
        with pytest.raises(ValueError, match="axis_names must be unique"):
            DeviceMesh(
                devices=cpu_devices(4),
                mesh_shape=(2, 2),
                axis_names=("tp", "tp"),
            )

    def test_axis_size_by_name(self) -> None:
        mesh = mesh_2d(2, 4)
        assert mesh.axis_size("dp") == 2
        assert mesh.axis_size("tp") == 4

    def test_axis_size_by_index(self) -> None:
        mesh = mesh_2d(2, 4)
        assert mesh.axis_size(0) == 2
        assert mesh.axis_size(1) == 4

    def test_resolve_axis_bad_name_raises(self) -> None:
        mesh = mesh_1d(4)
        with pytest.raises(ValueError, match="Unknown axis name 'bad'"):
            mesh.axis_size("bad")

    def test_resolve_axis_bad_index_raises(self) -> None:
        mesh = mesh_1d(4)
        with pytest.raises(IndexError, match="out of range"):
            mesh.axis_size(5)

    def test_resolve_axis_negative_index_raises(self) -> None:
        mesh = mesh_1d(4)
        with pytest.raises(IndexError, match="out of range"):
            mesh.axis_size(-1)

    def test_repr(self) -> None:
        mesh = mesh_2d(2, 4)
        assert repr(mesh) == "DeviceMesh(dp=2, tp=4)"

    def test_zero_devices_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one device"):
            DeviceMesh(devices=(), mesh_shape=(0,), axis_names=("tp",))

    def test_frozen(self) -> None:
        mesh = mesh_1d(4)
        with pytest.raises(dataclasses.FrozenInstanceError):
            mesh.mesh_shape = (2,)  # type: ignore[misc]
