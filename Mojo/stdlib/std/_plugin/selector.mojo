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
"""Shared compile-time plugin registration and selection utilities."""

from std.builtin.rebind import downcast
from std.builtin.variadics import TypeList
from std.sys.info import CompilationTarget, _TargetType, _current_target


trait Plugin:
    """Defines the identity required by a compile-time plugin registry.

    Each plugin system refines this trait with a parametric member
    `comptime HookFor[target: CompilationTarget]: <Hooks>` naming the hooks it
    uses for `target`. Traits can't take parameters, so this indirection is how
    a plugin's hook trait gets specialized per target: the plugin is a plain
    struct that selects a backend by `name`, and `HookFor[target]` is a
    separate struct parameterized on the target that implements the actual
    hooks.
    """

    comptime name: __mlir_type.`!kgen.string`
    """The stable identifier matched against a target's plugin field."""


struct PluginSelector[
    Trait: type_of(Plugin),
    //,
    *PluginTypes: Trait,
]:
    """Selects one of `PluginTypes` by matching a target's `stdlib_plugin`
    field against each plugin's `name`.

    The selected plugin is not yet specialized to a target; callers apply it
    with `HookFor`, for example
    `SELECTOR.current.HookFor[CompilationTarget.current()]` or
    `SELECTOR.for_target[target].HookFor[CompilationTarget[_mlir_value=target]()]`.
    """

    comptime index_for_target[
        target: _TargetType
    ]: __mlir_type.index = Self._find[target, __mlir_attr.`0 : index`]()

    comptime current: Self.Trait = Self.PluginTypes._get_type_at_index[
        Self.index_for_target[_current_target()]
    ]

    comptime for_target[target: _TargetType]: Self.Trait = (
        Self.PluginTypes._get_type_at_index[Self.index_for_target[target]]
    )

    comptime _length = Self.PluginTypes._raw_length

    @staticmethod
    def _matches[target: _TargetType, idx: __mlir_type.index]() -> Bool:
        """Returns whether plugin `idx` matches `target`."""
        return __mlir_attr[
            `#kgen.param.identical<`,
            __mlir_attr[
                `#kgen.param.expr<target_get_field,`,
                target,
                `, "stdlib_plugin" : !kgen.string`,
                `> : !kgen.string`,
            ],
            `,`,
            Self.PluginTypes._get_type_at_index[idx].name,
            `> : !kgen.scalar<bool>`,
        ]

    @staticmethod
    def _find[
        target: _TargetType, idx: __mlir_type.index
    ]() -> __mlir_type.index:
        """Finds the plugin matching `target` through parameter recursion."""
        comptime if not _index_lt[idx, Self._length]():
            __mlir_op.`llvm.intr.trap`()
            return idx
        elif Self._matches[target, idx]():
            return idx
        else:
            return Self._find[
                target,
                __mlir_attr[
                    `#kgen.param.expr<add,`,
                    idx,
                    `, 1 : index> : index`,
                ],
            ]()


# TODO: Inline this logic onto CompilationTarget
struct TargetPluginSelector[
    Trait: type_of(AnyType),
    //,
    target: CompilationTarget,
    Default: Trait,
]:
    """Selects the first plugin accepted by `conforms_to(P, Trait)`.

    The implementation takes care to minimize use of standard library types like
    `Int`, to avoid recursive resolution issues when resolving standard library
    types that themselves depend on resolving the current standard library
    plugin.
    """

    comptime _plugin_list_mlir_type = TypeList[Trait=AnyType]._mlir_type

    comptime selected: Self.Trait = (
        Self.Default if not _index_lt[
            Self._index, Self.Plugins._raw_length
        ]() else downcast[
            Self.Plugins._get_type_at_index[Self._index], Self.Trait
        ]
    )
    """Yields the matching plugin implementation of `Trait` for `target` if one
    is defined, otherwise yields `Default`."""

    comptime Plugins = TypeList[
        Trait=AnyType,
        __mlir_attr[
            `#kgen.get_target_plugin<`,
            Self.target._mlir_value,
            `, :`,
            +Self._plugin_list_mlir_type,
            ` `,
            TypeList.of[Trait=AnyType, Self.Default]().values,
            `> : `,
            +Self._plugin_list_mlir_type,
        ],
    ]()

    comptime _index = Self._find[__mlir_attr.`0 : index`]()

    @staticmethod
    def _find[idx: __mlir_type.index]() -> __mlir_type.index:
        comptime if not _index_lt[idx, Self.Plugins._raw_length]():
            __mlir_op.`llvm.intr.trap`()
            return idx
        elif conforms_to(Self.Plugins._get_type_at_index[idx], Self.Trait):
            return idx
        else:
            return Self._find[
                __mlir_attr[
                    `#kgen.param.expr<add,`,
                    idx,
                    `, 1 : index> : index`,
                ]
            ]()


def _index_lt[lhs: __mlir_type.index, rhs: __mlir_type.index]() -> Bool:
    """Returns whether raw parameter index `lhs` is less than `rhs`."""
    return __mlir_op.`index.cmp`[pred=__mlir_attr.`#index.cmp_predicate<ult>`](
        lhs, rhs
    )
