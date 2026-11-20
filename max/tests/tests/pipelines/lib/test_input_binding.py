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
"""Checks a role's shared graph and runtime input binding."""

from collections import OrderedDict
from unittest import TestCase

from max import tree
from max.pipelines.lib.input_binding import _flatten_role_inputs


@tree.dataclass(frozen=True)
class _Pair:
    first: object
    second: object


@tree.dataclass(frozen=True, kw_only=True)
class _Role:
    tokens: object = None
    kv: object = None
    state: object = None
    ep: object = None
    vision: object = None


class RoleInputPackingTest(TestCase):
    def test_the_graph_field_order_overrides_dataclass_declaration_order(
        self,
    ) -> None:
        _, structure = tree.flatten(
            OrderedDict(state=("s",), tokens="t", kv=_Pair("k", "v"))
        )
        payload = _Role(tokens=1, kv=_Pair(2, 3), state=(4,))
        self.assertEqual(
            _flatten_role_inputs(structure, payload, strict=True), (4, 1, 2, 3)
        )

    def test_the_role_dataclass_defines_the_shared_tree(self) -> None:
        types = _Role(tokens="t", kv=_Pair("k", "v"))
        _, structure = tree.flatten(types)
        values = _Role(tokens=1, kv=_Pair(2, 3))
        self.assertEqual(
            _flatten_role_inputs(structure, values, strict=True), (1, 2, 3)
        )

    def test_nested_state_uses_the_native_dataclass_structure(self) -> None:
        _, structure = tree.flatten(_Role(state=_Pair(("s0", "s1"), None)))
        runtime = _Role(state=_Pair((1, 2), None))
        self.assertEqual(
            _flatten_role_inputs(structure, runtime, strict=True), (1, 2)
        )

    def test_the_declared_mapping_order_is_preserved(self) -> None:
        _, structure = tree.flatten(
            OrderedDict(ep=OrderedDict(z="z-type", a=_Pair("a0", "a1")))
        )
        self.assertEqual(
            _flatten_role_inputs(
                structure,
                _Role(ep=OrderedDict(z=2, a=_Pair(3, 4))),
                strict=True,
            ),
            (2, 3, 4),
        )

    def test_runtime_mapping_order_cannot_reorder_graph_inputs(self) -> None:
        _, structure = tree.flatten(
            OrderedDict(ep=OrderedDict(z="z-type", a=_Pair("a0", "a1")))
        )
        for ep in (
            {"a": _Pair(3, 4), "z": 2},
            OrderedDict(a=_Pair(3, 4), z=2),
        ):
            with self.subTest(ep=ep), self.assertRaises(ValueError):
                _flatten_role_inputs(structure, _Role(ep=ep), strict=True)

    def test_unbound_optional_fields_are_omitted(self) -> None:
        _, structure = tree.flatten(OrderedDict(tokens="type"))
        self.assertEqual(
            _flatten_role_inputs(structure, _Role(tokens=1, ep=(2, 3))),
            (1,),
        )

    def test_runtime_fields_and_nested_keys_must_exist(self) -> None:
        _, structure = tree.flatten(OrderedDict(kv=OrderedDict(k="k", v="v")))
        cases: tuple[dict[str, object], ...] = (
            {},
            {"kv": None},
            {"kv": {"k": 1}},
            {"kv": {}},
        )
        for fields in cases:
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                _flatten_role_inputs(structure, fields)

    def test_extra_runtime_subtree_keys_are_rejected(self) -> None:
        _, structure = tree.flatten(OrderedDict(kv={"k": "k"}))
        with self.assertRaisesRegex(ValueError, "expected keys"):
            _flatten_role_inputs(structure, {"kv": {"k": 1, "v": 2}})

    def test_distributed_leaves_expand_into_physical_buffers(self) -> None:
        _, structure = tree.flatten(
            OrderedDict(vision=_Pair("embeddings", "indices"))
        )
        self.assertEqual(
            _flatten_role_inputs(structure, {"vision": _Pair((1, 2), (3, 4))}),
            (1, 2, 3, 4),
        )

    def test_strict_runtime_packing_rejects_short_or_long_payloads(
        self,
    ) -> None:
        _, structure = tree.flatten(OrderedDict(tokens="t", state=("s",)))
        for state in ((), (2, 3)):
            with self.subTest(state=state), self.assertRaises(ValueError):
                _flatten_role_inputs(
                    structure, {"tokens": 1, "state": state}, strict=True
                )

    def test_a_resource_provider_can_supply_an_explicit_flat_payload(
        self,
    ) -> None:
        _, structure = tree.flatten(OrderedDict(ep=_Pair("send", "receive")))
        with self.assertRaises(ValueError):
            _flatten_role_inputs(structure, {"ep": (1, 2)}, strict=True)
        self.assertEqual(
            _flatten_role_inputs(
                structure, {"ep": (1, 2)}, flat_fields=("ep",), strict=True
            ),
            (1, 2),
        )

    def test_a_flat_provider_can_expand_distributed_layout_leaves(self) -> None:
        _, structure = tree.flatten(OrderedDict(ep=_Pair("send", "receive")))
        payload = _Role(ep=(1, 2, 3, 4))
        self.assertEqual(
            _flatten_role_inputs(structure, payload, flat_fields=("ep",)),
            (1, 2, 3, 4),
        )
        with self.assertRaisesRegex(ValueError, "takes 2 input leaves"):
            _flatten_role_inputs(
                structure, payload, flat_fields=("ep",), strict=True
            )
