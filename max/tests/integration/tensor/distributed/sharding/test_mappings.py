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
"""Tests for :class:`DeviceMapping` and the :class:`NamedMapping` constructor.

After the mappings unification, ``DeviceMapping`` is a single concrete
class holding ``mesh + placements``. ``NamedMapping`` is sugar that
translates a JAX-style ``("dp", "tp", None)`` spec into placements at
construction time; the spec is *not* retained. Mesh-axis-name lookup
happens during construction via :func:`default_mesh` so a mapping built
inside ``with default_mesh(real_mesh):`` resolves against ``real_mesh``
even if the caller passed the singleton placeholder.
"""

from __future__ import annotations

import dataclasses

import pytest
from max.driver import CPU, Device
from max.experimental.sharding import (
    ConversionError,
    DeviceMapping,
    DeviceMesh,
    NamedMapping,
    Partial,
    PlacementMapping,
    Replicated,
    Sharded,
    is_fully_replicated,
)


def cpu_devices(n: int) -> tuple[Device, ...]:
    return tuple(CPU() for _ in range(n))


def mesh_1d(n: int, name: str = "tp") -> DeviceMesh:
    return DeviceMesh(cpu_devices(n), (n,), (name,))


def mesh_2d(rows: int, cols: int) -> DeviceMesh:
    return DeviceMesh(cpu_devices(rows * cols), (rows, cols), ("dp", "tp"))


class TestDeviceMapping:
    def test_construction(self) -> None:
        mesh = mesh_1d(4)
        m = DeviceMapping(mesh, (Sharded(0),))
        assert m.mesh is mesh
        assert m.placements == (Sharded(0),)

    def test_wrong_placement_count_raises(self) -> None:
        mesh = mesh_1d(4)
        with pytest.raises(ValueError, match="one placement per mesh axis"):
            DeviceMapping(mesh, (Sharded(0), Replicated()))

    def test_2d_mesh(self) -> None:
        mesh = mesh_2d(2, 4)
        m = DeviceMapping(mesh, (Replicated(), Sharded(1)))
        assert m.placements == (Replicated(), Sharded(1))

    def test_to_placements_alias(self) -> None:
        mesh = mesh_2d(2, 4)
        placements = (Sharded(0), Replicated())
        m = DeviceMapping(mesh, placements)
        assert m.to_placements() == placements

    def test_repr(self) -> None:
        mesh = mesh_1d(4)
        m = DeviceMapping(mesh, (Sharded(0),))
        r = repr(m)
        assert "DeviceMapping" in r
        assert "Sharded" in r

    def test_frozen(self) -> None:
        mesh = mesh_1d(4)
        m = DeviceMapping(mesh, (Sharded(0),))
        with pytest.raises(dataclasses.FrozenInstanceError):
            m.placements = ()  # type: ignore[misc]

    def test_placement_mapping_is_alias(self) -> None:
        assert PlacementMapping is DeviceMapping


class TestIsFullyReplicated:
    def test_all_replicated(self) -> None:
        mesh = mesh_2d(2, 4)
        m = DeviceMapping(mesh, (Replicated(), Replicated()))
        assert is_fully_replicated(m)

    def test_with_sharded(self) -> None:
        mesh = mesh_1d(4)
        m = DeviceMapping(mesh, (Sharded(0),))
        assert not is_fully_replicated(m)

    def test_with_partial(self) -> None:
        mesh = mesh_1d(4)
        m = DeviceMapping(mesh, (Partial(),))
        assert not is_fully_replicated(m)

    def test_single_device_mesh(self) -> None:
        mesh = DeviceMesh.single(CPU())
        m = DeviceMapping(mesh, (Replicated(),))
        assert is_fully_replicated(m)


class TestNamedMapping:
    """NamedMapping is a thin spec→placements constructor.

    After construction, every NamedMapping instance is structurally a
    DeviceMapping. The spec is *not* preserved.
    """

    def test_is_device_mapping(self) -> None:
        mesh = mesh_1d(4)
        ns = NamedMapping(mesh, ("tp", None))
        assert isinstance(ns, DeviceMapping)

    def test_basic_spec(self) -> None:
        mesh = mesh_1d(4)
        ns = NamedMapping(mesh, ("tp", None))
        assert ns.placements == (Sharded(0),)

    def test_2d_spec(self) -> None:
        mesh = mesh_2d(2, 4)
        ns = NamedMapping(mesh, ("dp", "tp"))
        assert ns.placements == (Sharded(0), Sharded(1))

    def test_all_none_replicated(self) -> None:
        mesh = mesh_1d(4)
        ns = NamedMapping(mesh, (None, None))
        assert ns.placements == (Replicated(),)
        assert is_fully_replicated(ns)

    def test_unknown_axis_drops_to_replicated(self) -> None:
        mesh = mesh_1d(4)
        ns = NamedMapping(mesh, ("bad", None))
        assert ns.placements == (Replicated(),)

    def test_unreduced_becomes_partial(self) -> None:
        mesh = mesh_1d(4)
        ns = NamedMapping(mesh, (None, None), unreduced=("tp",))
        assert ns.placements == (Partial(),)

    def test_unknown_unreduced_filtered(self) -> None:
        mesh = mesh_1d(4)
        ns = NamedMapping(mesh, (None, None), unreduced=("bad",))
        assert ns.placements == (Replicated(),)

    def test_shard_and_unreduced_overlap_raises(self) -> None:
        mesh = mesh_1d(4)
        with pytest.raises(ConversionError, match="both unreduced and used"):
            NamedMapping(mesh, ("tp", None), unreduced=("tp",))

    def test_axis_conflict_raises(self) -> None:
        mesh = mesh_1d(4)
        with pytest.raises(ConversionError, match="already assigned"):
            NamedMapping(mesh, ("tp", "tp"))

    def test_multi_axis_spec(self) -> None:
        mesh = mesh_2d(2, 4)
        ns = NamedMapping(mesh, (("dp", "tp"), None))
        assert ns.placements == (Sharded(0), Sharded(0))

    def test_shard_and_unreduced_different_axes(self) -> None:
        mesh = mesh_2d(2, 4)
        ns = NamedMapping(mesh, ("tp", None), unreduced=("dp",))
        assert ns.placements == (Partial(), Sharded(0))


class TestNamedMappingRepr:
    """``NamedMapping.__repr__`` renders placements in spec form."""

    def test_renders_sharded(self) -> None:
        mesh = mesh_2d(2, 4)
        ns = NamedMapping(mesh, ("dp", "tp"))
        r = repr(ns)
        assert "NamedMapping" in r
        assert "'dp'" in r and "'tp'" in r

    def test_renders_unreduced(self) -> None:
        mesh = mesh_2d(2, 4)
        ns = NamedMapping(mesh, ("tp", None), unreduced=("dp",))
        r = repr(ns)
        assert "unreduced" in r
        assert "'dp'" in r

    def test_renders_all_replicated(self) -> None:
        mesh = mesh_1d(4)
        ns = NamedMapping(mesh, (None,))
        # No sharded entries → empty spec
        assert "NamedMapping" in repr(ns)


class TestConversionError:
    def test_is_exception(self) -> None:
        assert issubclass(ConversionError, Exception)
