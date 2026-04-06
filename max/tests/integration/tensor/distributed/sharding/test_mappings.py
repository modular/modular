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
"""Tests for PlacementMapping, NamedMapping, DeviceMapping ABC,
roundtrip conversions, ConversionError, and edge cases."""

from __future__ import annotations

import dataclasses

import pytest
from max.driver import CPU, Device
from max.experimental.sharding import (
    ConversionError,
    DeviceMesh,
    NamedMapping,
    Partial,
    PlacementMapping,
    Replicated,
    Sharded,
)

# ── Inline mesh helpers (no conftest dependency) ──────────────────────


def cpu_devices(n: int) -> tuple[Device, ...]:
    return tuple(CPU() for _ in range(n))


def mesh_1d(n: int, name: str = "tp") -> DeviceMesh:
    return DeviceMesh(cpu_devices(n), (n,), (name,))


def mesh_2d(rows: int, cols: int) -> DeviceMesh:
    return DeviceMesh(cpu_devices(rows * cols), (rows, cols), ("dp", "tp"))


# ═════════════════════════════════════════════════════════════════════════
#  PlacementMapping
# ═════════════════════════════════════════════════════════════════════════


class TestPlacementMapping:
    def test_construction(self) -> None:
        mesh = mesh_1d(4)
        pm = PlacementMapping(mesh, (Sharded(0),))
        assert pm.mesh is mesh
        assert pm.placements == (Sharded(0),)

    def test_wrong_placement_count_raises(self) -> None:
        mesh = mesh_1d(4)
        with pytest.raises(ValueError, match="one placement per mesh axis"):
            PlacementMapping(mesh, (Sharded(0), Replicated()))

    def test_2d_mesh(self) -> None:
        mesh = mesh_2d(2, 4)
        pm = PlacementMapping(mesh, (Replicated(), Sharded(1)))
        assert pm.placements == (Replicated(), Sharded(1))

    def test_is_fully_resolved(self) -> None:
        mesh = mesh_1d(4)
        pm = PlacementMapping(mesh, (Sharded(0),))
        assert pm.is_fully_resolved

    def test_is_fully_replicated_true(self) -> None:
        mesh = mesh_2d(2, 4)
        pm = PlacementMapping(mesh, (Replicated(), Replicated()))
        assert pm.is_fully_replicated

    def test_is_fully_replicated_false(self) -> None:
        mesh = mesh_1d(4)
        pm = PlacementMapping(mesh, (Sharded(0),))
        assert not pm.is_fully_replicated

    def test_to_placements(self) -> None:
        mesh = mesh_2d(2, 4)
        placements = (Sharded(0), Replicated())
        pm = PlacementMapping(mesh, placements)
        assert pm.to_placements() == placements

    def test_repr(self) -> None:
        mesh = mesh_1d(4)
        pm = PlacementMapping(mesh, (Sharded(0),))
        r = repr(pm)
        assert "PlacementMapping" in r
        assert "Sharded" in r

    def test_frozen(self) -> None:
        mesh = mesh_1d(4)
        pm = PlacementMapping(mesh, (Sharded(0),))
        with pytest.raises(dataclasses.FrozenInstanceError):
            pm._placements = ()  # type: ignore[misc]

    def test_to_named_sharding_basic(self) -> None:
        mesh = mesh_1d(4)
        pm = PlacementMapping(mesh, (Sharded(0),))
        ns = pm.to_named_sharding(tensor_rank=2)
        assert isinstance(ns, NamedMapping)
        assert ns.spec == ("tp", None)

    def test_to_named_sharding_replicated(self) -> None:
        mesh = mesh_1d(4)
        pm = PlacementMapping(mesh, (Replicated(),))
        ns = pm.to_named_sharding(tensor_rank=2)
        assert ns.spec == (None, None)

    def test_to_named_sharding_partial(self) -> None:
        mesh = mesh_1d(4)
        pm = PlacementMapping(mesh, (Partial(),))
        ns = pm.to_named_sharding(tensor_rank=2)
        assert ns.spec == (None, None)
        assert ns.unreduced == frozenset({"tp"})

    def test_to_named_sharding_2d(self) -> None:
        mesh = mesh_2d(2, 4)
        pm = PlacementMapping(mesh, (Sharded(0), Sharded(1)))
        ns = pm.to_named_sharding(tensor_rank=2)
        assert ns.spec == ("dp", "tp")

    def test_to_named_sharding_out_of_range_raises(self) -> None:
        mesh = mesh_1d(4)
        pm = PlacementMapping(mesh, (Sharded(5),))
        with pytest.raises(ConversionError, match="out of range"):
            pm.to_named_sharding(tensor_rank=2)

    def test_to_named_sharding_multi_axis_same_dim(self) -> None:
        mesh = mesh_2d(2, 4)
        pm = PlacementMapping(mesh, (Sharded(0), Sharded(0)))
        ns = pm.to_named_sharding(tensor_rank=2)
        assert ns.spec == (("dp", "tp"), None)


# ═════════════════════════════════════════════════════════════════════════
#  NamedMapping
# ═════════════════════════════════════════════════════════════════════════


class TestNamedMapping:
    def test_construction(self) -> None:
        mesh = mesh_1d(4)
        ns = NamedMapping(mesh, ("tp", None))
        assert ns.mesh is mesh
        assert ns.spec == ("tp", None)

    def test_bad_axis_name_raises(self) -> None:
        mesh = mesh_1d(4)
        with pytest.raises(ValueError, match="Unknown mesh axis 'bad'"):
            NamedMapping(mesh, ("bad", None))

    def test_bad_unreduced_axis_raises(self) -> None:
        mesh = mesh_1d(4)
        with pytest.raises(ValueError, match="Unknown mesh axis 'bad'"):
            NamedMapping(mesh, (None, None), _unreduced=frozenset({"bad"}))

    def test_priorities_length_mismatch_raises(self) -> None:
        mesh = mesh_1d(4)
        with pytest.raises(ValueError, match="priorities length"):
            NamedMapping(mesh, ("tp", None), _priorities=(0,))

    def test_shard_and_unreduced_overlap_raises(self) -> None:
        mesh = mesh_1d(4)
        with pytest.raises(ValueError, match="both sharding and unreduced"):
            NamedMapping(mesh, ("tp", None), _unreduced=frozenset({"tp"}))

    def test_is_fully_replicated_true(self) -> None:
        mesh = mesh_1d(4)
        ns = NamedMapping(mesh, (None, None))
        assert ns.is_fully_replicated

    def test_is_fully_replicated_false_sharded(self) -> None:
        mesh = mesh_1d(4)
        ns = NamedMapping(mesh, ("tp", None))
        assert not ns.is_fully_replicated

    def test_is_fully_replicated_false_unreduced(self) -> None:
        mesh = mesh_1d(4)
        ns = NamedMapping(mesh, (None, None), _unreduced=frozenset({"tp"}))
        assert not ns.is_fully_replicated

    def test_is_fully_resolved_true(self) -> None:
        mesh = mesh_1d(4)
        ns = NamedMapping(mesh, ("tp", None))
        assert ns.is_fully_resolved

    def test_is_fully_resolved_false_priorities(self) -> None:
        mesh = mesh_1d(4)
        ns = NamedMapping(mesh, ("tp", None), _priorities=(0, 1))
        assert not ns.is_fully_resolved

    def test_to_placements_basic(self) -> None:
        mesh = mesh_1d(4)
        ns = NamedMapping(mesh, ("tp", None))
        assert ns.to_placements() == (Sharded(0),)

    def test_to_placements_2d(self) -> None:
        mesh = mesh_2d(2, 4)
        ns = NamedMapping(mesh, ("dp", "tp"))
        assert ns.to_placements() == (Sharded(0), Sharded(1))

    def test_to_placements_replicated(self) -> None:
        mesh = mesh_1d(4)
        ns = NamedMapping(mesh, (None, None))
        assert ns.to_placements() == (Replicated(),)

    def test_to_placements_unreduced(self) -> None:
        mesh = mesh_1d(4)
        ns = NamedMapping(mesh, (None, None), _unreduced=frozenset({"tp"}))
        assert ns.to_placements() == (Partial(),)

    def test_to_placements_priorities_raises(self) -> None:
        mesh = mesh_1d(4)
        ns = NamedMapping(mesh, ("tp", None), _priorities=(0, 1))
        with pytest.raises(ConversionError, match="priorities"):
            ns.to_placements()

    def test_to_placements_axis_conflict_raises(self) -> None:
        # Two tensor dims both mapped to the same mesh axis
        mesh = mesh_1d(4)
        ns = NamedMapping(mesh, ("tp", "tp"))
        with pytest.raises(ConversionError, match="already assigned"):
            ns.to_placements()

    def test_to_named_sharding_returns_self(self) -> None:
        mesh = mesh_1d(4)
        ns = NamedMapping(mesh, ("tp", None))
        assert ns.to_named_sharding(tensor_rank=2) is ns

    def test_repr(self) -> None:
        mesh = mesh_1d(4)
        ns = NamedMapping(mesh, ("tp", None))
        r = repr(ns)
        assert "NamedMapping" in r
        assert "'tp'" in r

    def test_repr_with_unreduced(self) -> None:
        mesh = mesh_2d(2, 4)
        ns = NamedMapping(mesh, (None, "tp"), _unreduced=frozenset({"dp"}))
        r = repr(ns)
        assert "unreduced" in r

    def test_memory_kind(self) -> None:
        mesh = mesh_1d(4)
        ns = NamedMapping(mesh, (None,), _memory_kind="device")
        assert ns.memory_kind == "device"
        assert "memory_kind" in repr(ns)

    def test_multi_axis_spec(self) -> None:
        mesh = mesh_2d(2, 4)
        ns = NamedMapping(mesh, (("dp", "tp"), None))
        placements = ns.to_placements()
        assert placements == (Sharded(0), Sharded(0))


# ═════════════════════════════════════════════════════════════════════════
#  Roundtrip: PlacementMapping <-> NamedMapping
# ═════════════════════════════════════════════════════════════════════════


class TestRoundtrip:
    def test_placement_to_named_to_placement(self) -> None:
        mesh = mesh_2d(2, 4)
        original = (Sharded(0), Replicated())
        pm = PlacementMapping(mesh, original)
        ns = pm.to_named_sharding(tensor_rank=3)
        roundtripped = ns.to_placements()
        assert roundtripped == original

    def test_named_to_placement_to_named(self) -> None:
        mesh = mesh_2d(2, 4)
        ns = NamedMapping(mesh, ("dp", "tp", None))
        placements = ns.to_placements()
        pm = PlacementMapping(mesh, placements)
        ns2 = pm.to_named_sharding(tensor_rank=3)
        assert ns2.spec == ns.spec


# ═════════════════════════════════════════════════════════════════════════
#  ConversionError
# ═════════════════════════════════════════════════════════════════════════


class TestConversionError:
    def test_is_exception(self) -> None:
        assert issubclass(ConversionError, Exception)

    def test_message(self) -> None:
        err = ConversionError("test message")
        assert str(err) == "test message"


# ═════════════════════════════════════════════════════════════════════════
#  PlacementMapping edge cases
# ═════════════════════════════════════════════════════════════════════════


class TestPlacementMappingEdgeCases:
    def test_is_fully_replicated_false_with_partial(self) -> None:
        mesh = mesh_1d(4)
        pm = PlacementMapping(mesh, (Partial(),))
        assert not pm.is_fully_replicated

    def test_single_device_mesh(self) -> None:
        from max.driver import CPU

        mesh = DeviceMesh.single(CPU())
        pm = PlacementMapping(mesh, (Replicated(),))
        assert pm.is_fully_replicated
        assert pm.is_fully_resolved

    def test_to_named_sharding_with_partial_and_shard(self) -> None:
        mesh = mesh_2d(2, 4)
        pm = PlacementMapping(mesh, (Partial(), Sharded(1)))
        ns = pm.to_named_sharding(tensor_rank=3)
        assert ns.spec == (None, "tp", None)
        assert ns.unreduced == frozenset({"dp"})


# ═════════════════════════════════════════════════════════════════════════
#  DeviceMapping ABC
# ═════════════════════════════════════════════════════════════════════════


class TestDeviceMapping:
    """PlacementMapping and NamedMapping both implement DeviceMapping."""

    def test_placement_mapping_is_device_mapping(self) -> None:
        from max.experimental.sharding import DeviceMapping

        mesh = mesh_1d(4)
        pm = PlacementMapping(mesh, (Sharded(0),))
        assert isinstance(pm, DeviceMapping)

    def test_named_mapping_is_device_mapping(self) -> None:
        from max.experimental.sharding import DeviceMapping

        mesh = mesh_1d(4)
        ns = NamedMapping(mesh, ("tp", None))
        assert isinstance(ns, DeviceMapping)

    def test_device_mapping_protocol_mesh(self) -> None:
        from max.experimental.sharding import DeviceMapping

        mesh = mesh_1d(4)
        pm: DeviceMapping = PlacementMapping(mesh, (Sharded(0),))
        assert pm.mesh is mesh

    def test_device_mapping_protocol_is_fully_resolved(self) -> None:
        from max.experimental.sharding import DeviceMapping

        mesh = mesh_1d(4)
        pm: DeviceMapping = PlacementMapping(mesh, (Sharded(0),))
        assert pm.is_fully_resolved

        ns: DeviceMapping = NamedMapping(mesh, ("tp", None), _priorities=(0, 1))
        assert not ns.is_fully_resolved


# ═════════════════════════════════════════════════════════════════════════
#  NamedMapping with both sharding and unreduced (different axes)
# ═════════════════════════════════════════════════════════════════════════


class TestNamedMappingMixed:
    def test_shard_and_unreduced_different_axes(self) -> None:
        mesh = mesh_2d(2, 4)
        ns = NamedMapping(mesh, ("tp", None), _unreduced=frozenset({"dp"}))
        placements = ns.to_placements()
        assert placements == (Partial(), Sharded(0))

    def test_is_fully_replicated_with_sharding(self) -> None:
        mesh = mesh_2d(2, 4)
        ns = NamedMapping(mesh, ("tp", None), _unreduced=frozenset({"dp"}))
        assert not ns.is_fully_replicated

    def test_roundtrip_partial_and_shard(self) -> None:
        mesh = mesh_2d(2, 4)
        pm = PlacementMapping(mesh, (Partial(), Sharded(1)))
        ns = pm.to_named_sharding(tensor_rank=3)
        assert ns.unreduced == frozenset({"dp"})
        assert ns.spec == (None, "tp", None)
        # Roundtrip back
        assert ns.to_placements() == (Partial(), Sharded(1))
