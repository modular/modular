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
"""Tests for DeviceMesh construction, properties, and equality."""

from __future__ import annotations

import dataclasses

import pytest
from max.driver import CPU, Device
from max.experimental.sharding import DeviceMesh

# ── Inline mesh helpers (no conftest dependency) ──────────────────────


def cpu_devices(n: int) -> tuple[Device, ...]:
    return tuple(CPU() for _ in range(n))


def mesh_1d(n: int, name: str = "tp") -> DeviceMesh:
    return DeviceMesh(cpu_devices(n), (n,), (name,))


def mesh_2d(rows: int, cols: int) -> DeviceMesh:
    return DeviceMesh(cpu_devices(rows * cols), (rows, cols), ("dp", "tp"))


# ═════════════════════════════════════════════════════════════════════════
#  DeviceMesh
# ═════════════════════════════════════════════════════════════════════════


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

    def test_single(self) -> None:
        from max.driver import CPU

        mesh = DeviceMesh.single(CPU())
        assert mesh.num_devices == 1
        assert mesh.is_single
        assert mesh.ndim == 1

    def test_is_single_false_for_multi(self) -> None:
        mesh = mesh_1d(4)
        assert not mesh.is_single

    def test_simulated_mesh_same_devices(self) -> None:
        cpu = CPU()
        mesh = DeviceMesh(
            devices=(cpu, cpu, cpu, cpu),
            mesh_shape=(4,),
            axis_names=("tp",),
        )
        assert mesh.is_simulated

    def test_is_simulated_false_for_single(self) -> None:
        mesh = DeviceMesh.single(CPU())
        assert not mesh.is_simulated


# ═════════════════════════════════════════════════════════════════════════
#  DeviceMesh equality (frozen dataclass)
# ═════════════════════════════════════════════════════════════════════════


class TestDeviceMeshEquality:
    def test_same_devices_equal(self) -> None:
        a = mesh_1d(4)
        b = mesh_1d(4)
        assert a == b

    def test_different_shape_not_equal(self) -> None:
        a = mesh_1d(4)
        b = mesh_2d(2, 2)
        assert a != b

    def test_different_names_not_equal(self) -> None:
        devs = cpu_devices(4)
        a = DeviceMesh(devs, (4,), ("tp",))
        b = DeviceMesh(devs, (4,), ("dp",))
        assert a != b

    def test_hashable(self) -> None:
        a = mesh_1d(4)
        b = mesh_1d(4)
        assert hash(a) == hash(b)
        assert {a, b} == {a}

    def test_mixed_mesh_raises(self) -> None:
        """Rejects meshes with some repeated and some different devices."""
        try:
            from max.driver import Accelerator

            devices = (
                Accelerator(0),
                Accelerator(0),
                Accelerator(1),
            )
        except Exception:
            pytest.skip("Need at least 2 GPUs for mixed-mesh test")
        with pytest.raises(ValueError, match=r"all-same.*or all-distinct"):
            DeviceMesh(
                devices=devices,
                mesh_shape=(3,),
                axis_names=("tp",),
            )
