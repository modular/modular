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
"""Encodes tensor layouts, with their device meshes and placements, as JSON.

A serialized :class:`~max.experimental.compilation.CompiledCallable` stores
its input and output layouts this way.
"""

from __future__ import annotations

from max._artifact.value_types import (
    ArtifactError,
    ArtifactType,
    expect,
    malformed,
)
from max._core.dialects import kgen
from max.driver import Device
from max.dtype import DType
from max.experimental.sharding import (
    BufferLayout,
    DeviceMapping,
    DeviceMesh,
    Partial,
    Placement,
    Replicated,
    Sharded,
    TensorLayout,
    Unknown,
)
from max.graph import AlgebraicDim, DeviceRef, Dim, StaticDim, SymbolicDim


def layout_to_json(layout: TensorLayout) -> dict[str, ArtifactType]:
    """Encodes a layout, with its device mesh and placements."""
    mesh = layout.mesh
    return {
        "buffer": isinstance(layout, BufferLayout),
        "dtype": layout.dtype.name,
        "shape": [_dim_to_json(dim, layout) for dim in layout.shape],
        "mesh": {
            "devices": [_device_to_json(device) for device in mesh.devices],
            "shape": list(mesh.mesh_shape),
            "axis_names": list(mesh.axis_names),
        },
        "placements": [_placement_to_json(p) for p in layout.placements],
    }


def _device_to_json(device: Device) -> dict[str, ArtifactType]:
    ref = DeviceRef.from_device(device)
    return {"type": ref.device_type.value, "id": ref.id}


def _placement_to_json(placement: Placement) -> dict[str, ArtifactType]:
    if isinstance(placement, Sharded):
        return {"type": "sharded", "axis": placement.axis}
    for name, kind in _PLACEMENTS.items():
        if type(placement) is kind:
            return {"type": name}
    raise ArtifactError(f"placement {placement!r} is not serializable")


# The placements that carry no fields, by serialized name.
_PLACEMENTS: dict[str, type[Placement]] = {
    "replicated": Replicated,
    "partial": Partial,
    "unknown": Unknown,
}


# The operators the graph API builds algebraic dims from.
_DIM_OPS = {
    "add": kgen.POC.add,
    "mul": kgen.POC.mul_no_wrap,
    "div": kgen.POC.div,
}


def _dim_to_json(dim: Dim, layout: TensorLayout) -> ArtifactType:
    """Encodes a dim as an integer, a symbol, or an operator over dims."""
    if isinstance(dim, StaticDim):
        return int(dim)
    if isinstance(dim, SymbolicDim):
        return dim.name
    if isinstance(dim, AlgebraicDim):
        for name, opcode in _DIM_OPS.items():
            if dim.attr.opcode == opcode:
                return {
                    "op": name,
                    "operands": [
                        _dim_to_json(Dim(operand), layout)
                        for operand in dim.attr.operands
                    ],
                }
    raise ArtifactError(f"dimension {dim} of {layout} is not serializable")


def _dim_from_json(value: ArtifactType) -> Dim:
    if isinstance(value, dict):
        op = expect(value["op"], str)
        if op not in _DIM_OPS:
            raise ValueError(f"unknown dimension operator {op!r}")
        operands = expect(value["operands"], list)
        return AlgebraicDim.apply(
            _DIM_OPS[op], *(_dim_from_json(operand) for operand in operands)
        )
    if isinstance(value, str):
        return Dim(value)
    return Dim(expect(value, int))


def layouts_from_json(value: ArtifactType, field: str) -> list[TensorLayout]:
    """Decodes the layouts :func:`layout_to_json` encoded.

    Args:
        value: The list of encoded layouts.
        field: The field ``value`` was read from, named in errors.

    Raises:
        ArtifactError: If ``value`` or one of its layouts is malformed.
    """
    try:
        items = expect(value, list)
    except TypeError as e:
        raise malformed(field, e) from e
    layouts: list[TensorLayout] = []
    for i, item in enumerate(items):
        try:
            layouts.append(_layout_from_json(item))
        except (KeyError, TypeError, ValueError) as e:
            raise malformed(f"{field}[{i}]", e) from e
    return layouts


def _layout_from_json(value: ArtifactType) -> TensorLayout:
    fields = expect(value, dict)
    dims = [_dim_from_json(dim) for dim in expect(fields["shape"], list)]
    mapping = DeviceMapping(
        _mesh_from_json(fields["mesh"]),
        tuple(
            _placement_from_json(placement)
            for placement in expect(fields["placements"], list)
        ),
    )
    kind = BufferLayout if expect(fields["buffer"], bool) else TensorLayout
    return kind(DType[expect(fields["dtype"], str)], dims, mapping)


def _mesh_from_json(value: ArtifactType) -> DeviceMesh:
    fields = expect(value, dict)
    devices = []
    for device in expect(fields["devices"], list):
        entry = expect(device, dict)
        ref = DeviceRef(expect(entry["type"], str), expect(entry["id"], int))
        devices.append(ref.to_device())
    mesh_shape = [expect(size, int) for size in expect(fields["shape"], list)]
    axis_names = [
        expect(name, str) for name in expect(fields["axis_names"], list)
    ]
    return DeviceMesh(tuple(devices), tuple(mesh_shape), tuple(axis_names))


def _placement_from_json(value: ArtifactType) -> Placement:
    fields = expect(value, dict)
    name = expect(fields["type"], str)
    if name == "sharded":
        return Sharded(expect(fields["axis"], int))
    if name not in _PLACEMENTS:
        raise ValueError(f"unknown placement {name!r}")
    return _PLACEMENTS[name]()
