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
construction time; the spec is *not* retained. Re-resolving against a
different mesh goes through :meth:`DeviceMapping.to_mesh`, which works
uniformly for any mapping by axis-name correspondence.
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
    Replicated,
    Sharded,
    mesh_context,
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

    def test_replicated(self) -> None:
        mesh = mesh_2d(2, 4)
        m = DeviceMapping.replicated(mesh)
        assert m.mesh is mesh
        assert m.placements == (Replicated(), Replicated())
        assert m.is_fully_replicated

    def test_replicated_single_axis(self) -> None:
        mesh = mesh_1d(4)
        assert DeviceMapping.replicated(mesh).placements == (Replicated(),)


class TestIsFullyReplicated:
    def test_all_replicated(self) -> None:
        mesh = mesh_2d(2, 4)
        m = DeviceMapping(mesh, (Replicated(), Replicated()))
        assert m.is_fully_replicated

    def test_with_sharded(self) -> None:
        mesh = mesh_1d(4)
        m = DeviceMapping(mesh, (Sharded(0),))
        assert not m.is_fully_replicated

    def test_with_partial(self) -> None:
        mesh = mesh_1d(4)
        m = DeviceMapping(mesh, (Partial(),))
        assert not m.is_fully_replicated

    def test_single_device_mesh(self) -> None:
        mesh = DeviceMesh.single(CPU())
        m = DeviceMapping(mesh, (Replicated(),))
        assert m.is_fully_replicated


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
        assert ns.is_fully_replicated

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


class TestActiveMesh:
    """``NamedMapping`` takes its mesh from ``mesh_context`` when given none.

    This is what lets a spec be written once, where the layer is defined, and
    resolved against whatever mesh the caller publishes -- so placement is a
    property of the enclosing context rather than a constructor argument
    threaded through every layer.
    """

    def test_takes_the_active_mesh(self) -> None:
        mesh = mesh_1d(4)
        with mesh_context(mesh):
            mapping = NamedMapping(spec=("tp",))
        assert mapping.mesh is mesh
        assert mapping.placements == (Sharded(0),)

    def test_explicit_mesh_wins(self) -> None:
        with mesh_context(mesh_1d(4)):
            mapping = NamedMapping(mesh_1d(2), ("tp",))
        assert mapping.mesh.num_devices == 2

    def test_same_spec_resolves_against_each_context(self) -> None:
        spec = ("tp", None)
        with mesh_context(mesh_1d(2)):
            two = NamedMapping(spec=spec)
        with mesh_context(mesh_1d(8)):
            eight = NamedMapping(spec=spec)
        assert two.mesh.num_devices == 2
        assert eight.mesh.num_devices == 8
        assert two.placements == eight.placements == (Sharded(0),)

    def test_an_axis_the_context_mesh_lacks_replicates(self) -> None:
        """The degradation that makes one source run on any topology."""
        with mesh_context(mesh_1d(4, name="other")):
            mapping = NamedMapping(spec=("tp", None))
        assert mapping.placements == (Replicated(),)

    def test_no_mesh_and_no_context_raises(self) -> None:
        with pytest.raises(ValueError, match="needs a mesh"):
            NamedMapping(spec=("tp",))
