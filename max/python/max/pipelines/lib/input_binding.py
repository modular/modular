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
"""Packs a role's physical buffers using its native tree structure."""

from __future__ import annotations

from collections.abc import Collection, Sequence

from max import tree
from max.driver import Buffer


def _physical_leaves(values: Sequence[object]) -> list[object]:
    """Returns ``tree.leaves(values)``.

    Buffers and flat tuples of per-device buffers skip the full tree walk,
    which costs several microseconds per input on every decode step.
    """
    leaves: list[object] = []
    for value in values:
        if isinstance(value, Buffer):
            leaves.append(value)
        elif (type(value) is tuple or type(value) is list) and all(
            isinstance(item, Buffer) for item in value
        ):
            leaves.extend(value)
        else:
            leaves.extend(tree.leaves(value))
    return leaves


def _flatten_role_inputs(
    input_tree_def: tree.TreeDef,
    inputs: object,
    *,
    flat_fields: Collection[str] = (),
    strict: bool = False,
) -> tuple[object, ...]:
    """Packs a runtime role dataclass in the graph's field order.

    Native tree operations own nesting and validation. Only physical buffer
    expansion and an already-flat resource payload need backend policy here.

    Args:
        input_tree_def: The graph's semantic input structure, in ABI order.
        inputs: The role's runtime ``tree.dataclass`` payload.
        flat_fields: Resource fields already packed in physical graph order.
        strict: Whether physical and declared leaf counts must match. V2
            requires this; V3 layouts can expand to several device buffers.

    Returns:
        The buffers in the graph's positional input order.

    Raises:
        ValueError: If a required field is missing, a subtree differs, or
            ``strict`` is true and the physical leaf count differs.
    """
    children, keys, _ = tree.flatten_one_level(inputs)
    if keys is None or (children and not keys):
        raise ValueError("runtime role inputs must have named fields")
    fields = dict(zip(keys, children, strict=True))
    buffers: list[object] = []
    for name, structure in zip(
        input_tree_def.child_keys, input_tree_def.children, strict=True
    ):
        if name not in fields or fields[name] is None:
            raise ValueError(f"no runtime value for role input {name!r}")
        field_inputs = (
            tree.leaves(fields[name])
            if name in flat_fields
            else structure.flatten_up_to(fields[name], name)
        )
        physical = _physical_leaves(field_inputs)
        if strict and len(physical) != structure.num_leaves:
            raise ValueError(
                f"{name}: the role takes {structure.num_leaves} input leaves, "
                f"got {len(physical)} runtime buffers"
            )
        buffers.extend(physical)
    return tuple(buffers)
