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
"""Cross-reference ``typing`` names against the Python inventory.

``autodoc_typehints_format`` defaults to "short", so type annotations
render ``typing.Any`` as bare ``Any``. The intersphinx inventory only
knows the qualified name, and the Python inventory additionally splits
``typing`` members across object types (``typing.Any`` is py:data,
``typing.Generic`` is py:class) while intersphinx maps a reference's
role to exactly one object type. The handlers here qualify bare names
where they mean ``typing`` members, and retry unresolved ``typing.*``
references with the object type the inventory itself uses. A
``:class:`` reference that still fails resolves as ``:obj:``, which
reaches type aliases documented with ``.. py:data::``.
"""

import collections.abc
import importlib
import typing

from docutils import nodes
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.environment import BuildEnvironment
from sphinx.ext.intersphinx import (
    InventoryAdapter,
    resolve_reference_in_inventory,
)

# The public typing API only; ``dir(typing)`` would also cover the
# module's own implementation imports (``sys``, ``types``, ``re``).
_typing_public = set(typing.__all__)


_abc_public = set(collections.abc.__all__)


def _stdlib_default(name: str) -> str:
    # The project imports container ABCs (``Sequence``, ``Callable``)
    # from ``collections.abc``; ``typing`` only re-exports deprecated
    # aliases of them.
    return "collections.abc" if name in _abc_public else "typing"


def _stdlib_module_for(module_name: str, name: str) -> str | None:
    """Return the stdlib module a bare *name* on *module_name*'s page means.

    ``None`` means the name is an in-project object and keeps its
    project reference.
    """
    try:
        module = importlib.import_module(module_name)
    except Exception:
        # Unimportable (for example an unbuildable extension): the
        # bare name carries no project meaning either way.
        return _stdlib_default(name)
    attr = getattr(module, name, None)
    if attr is None:
        # The page's module (often a package) doesn't import the name
        # that a member's defining module does.
        return _stdlib_default(name)
    attr_module = getattr(attr, "__module__", None)
    if attr_module == "typing":
        return "typing"
    if attr_module in ("collections.abc", "_collections_abc"):
        return "collections.abc"
    return None


def qualify_typing_xrefs(app: Sphinx, doctree: nodes.document) -> None:
    """Qualify bare ``typing`` and ``collections.abc`` names in place.

    A module whose same-named attribute is an in-project class
    (``Generator``, ``Type``) keeps its project reference.
    """
    for node in doctree.findall(addnodes.pending_xref):
        if node.get("refdomain") != "py":
            continue
        target = node.get("reftarget", "")
        if target not in _typing_public or not node.get("py:module"):
            continue
        stdlib_module = _stdlib_module_for(node["py:module"], target)
        if stdlib_module is not None:
            node["reftarget"] = f"{stdlib_module}.{target}"
            node["refspecific"] = False


def resolve_typing_inventory_links(
    app: Sphinx,
    env: BuildEnvironment,
    node: addnodes.pending_xref,
    contnode: nodes.Element,
) -> nodes.Element | None:
    """Retry a failed ``typing.*`` reference with the inventory's objtype."""
    if node.get("refdomain") != "py":
        return None
    target = node.get("reftarget", "")
    if not target.startswith("typing."):
        return None
    named_inventory = InventoryAdapter(env).named_inventory
    for inv_name, inventory in named_inventory.items():
        for objtype in ("py:data", "py:class"):
            if objtype not in inventory:
                continue
            if target not in inventory[objtype]:
                continue
            original_reftype = node["reftype"]
            node["reftype"] = objtype.split(":", 1)[1]
            try:
                resolved = resolve_reference_in_inventory(
                    env, inv_name, node, contnode
                )
            finally:
                node["reftype"] = original_reftype
            if resolved is not None:
                return resolved
    return None


def _resolve_class_as_obj(
    app: Sphinx,
    env: BuildEnvironment,
    node: addnodes.pending_xref,
    contnode: nodes.Element,
) -> nodes.Element | None:
    """Fall back to :obj: when a :class: cross-reference fails to resolve.

    Autodoc emits :class: references for all type annotations, but type
    aliases documented with ``.. py:data::`` are only registered as
    ``py:data`` targets.  Re-trying with :obj: (which matches any Python
    domain object type) lets those references resolve without forcing us
    to mis-label type aliases as classes in the RST source.
    """
    if node.get("refdomain") == "py" and node.get("reftype") == "class":
        return env.get_domain("py").resolve_xref(
            env,
            node.get("refdoc"),
            app.builder,
            "obj",
            node.get("reftarget"),
            node,
            contnode,
        )
    return None


def resolve_missing_py_reference(
    app: Sphinx,
    env: BuildEnvironment,
    node: addnodes.pending_xref,
    contnode: nodes.Element,
) -> nodes.Element | None:
    """Resolve a failed Python reference against the inventory, then as :obj:."""
    # A resolved node with no children is falsy, so test for ``None``.
    resolved = resolve_typing_inventory_links(app, env, node, contnode)
    if resolved is not None:
        return resolved
    return _resolve_class_as_obj(app, env, node, contnode)


def setup(app: Sphinx) -> None:
    app.connect("doctree-read", qualify_typing_xrefs)
    app.connect("missing-reference", resolve_missing_py_reference)
