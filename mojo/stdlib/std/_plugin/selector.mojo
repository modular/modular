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

from std.sys.info import _current_target

from ._overlay import PLUGINS


# Number of plugins, as a raw kgen `index`. Avoids `TypeList.size`'s `Int`
# wrapper so the selector never touches `Int`/`SIMDLength` comparison machinery.
comptime _PLUGIN_COUNT = __mlir_attr[
    `#kgen.param_list.size<:`,
    PLUGINS._mlir_type,
    ` `,
    +PLUGINS.values,
    `> : index`,
]


def _index_lt[lhs: __mlir_type.index, rhs: __mlir_type.index]() -> Bool:
    """`lhs < rhs` on raw `index`, via `index.cmp` (no `Int`/`SIMDLength`)."""
    return __mlir_op.`index.cmp`[pred=__mlir_attr.`#index<cmp_predicate ult>`](
        lhs, rhs
    )


def _plugin_matches[
    target: __mlir_type.`!kgen.target`, idx: __mlir_type.index
]() -> Bool:
    """Whether `PLUGINS[idx].name` equals `target`'s `stdlib_plugin` field."""
    return __mlir_attr[
        `#kgen.param.expr<eq,`,
        __mlir_attr[
            `#kgen.param.expr<target_get_field,`,
            target,
            `, "stdlib_plugin" : !kgen.string`,
            `> : !kgen.string`,
        ],
        `,`,
        PLUGINS._get_type_at_index[idx].name,
        `> : !kgen.scalar<bool>`,
    ]


def _find_plugin[
    target: __mlir_type.`!kgen.target`, idx: __mlir_type.index
]() -> __mlir_type.index:
    """Parameter-recursive scan of `PLUGINS` for the matching plugin index.

    Operates entirely on raw `index` (compare/increment via `index.*` ops) and
    uses parameter recursion rather than `comptime for`, so resolving the
    selector never instantiates `paramfor_has_next` or the `Int`/`SIMDLength`
    comparison machinery during stdlib bootstrap.
    """
    comptime if not _index_lt[idx, _PLUGIN_COUNT]():
        __mlir_op.`llvm.intr.trap`()
        return idx
    elif _plugin_matches[target, idx]():
        return idx
    else:
        return _find_plugin[
            target,
            __mlir_attr[
                `#kgen.param.expr<add,`,
                idx,
                `, 1 : index> : index`,
            ],
        ]()


def get_plugin_index[
    target: __mlir_type.`!kgen.target` = _current_target()
]() -> __mlir_type.index:
    """Returns the `index` into `PLUGINS` of the plugin matching `target`."""
    return _find_plugin[target, __mlir_attr.`0 : index`]()
