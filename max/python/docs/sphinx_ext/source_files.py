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
"""Locate source files, ``.pyi`` stubs, and symbol definitions.

``linkcode_resolve`` in conf.py and the docstring fallbacks in
:mod:`docstring_sources` both need to answer "where is this object
defined?" for objects whose runtime form carries no usable source
location: C-extension symbols, type aliases, constants, and names
re-exported through ``__init__.py`` chains. :func:`find_definition` walks
imports and ``.pyi`` stubs to find the defining module and line range.
"""

import ast
import functools
import importlib
import os
from typing import NamedTuple


def _resolve_relative_import(
    module_name: str, from_module: str | None, level: int, is_package: bool
) -> str | None:
    """Resolve a relative import to an absolute module name.

    For example, in package ``max.engine``, ``from .api import X`` (level=1,
    from_module='api') resolves to ``max.engine.api``.

    Args:
        module_name: Fully-qualified name of the module containing the import.
        from_module: The ``module`` field of the ``ImportFrom`` AST node
            (e.g. ``'api'`` for ``from .api import X``), or ``None`` for
            bare ``from . import X`` statements.
        level: Number of leading dots (1 = current package, 2 = parent, …).
        is_package: Whether *module_name* is a package (``__init__.py``).
            Packages treat level-1 as "self"; plain modules treat it as
            "parent".
    """
    if level == 0:
        return from_module
    parts = module_name.split(".")
    # Plain modules live *inside* their parent package, so strip the
    # module name first to get the package, then climb (level - 1) more.
    if not is_package:
        parts = parts[:-1]
    up = level - 1
    if up > len(parts):
        return None
    base = ".".join(parts[: len(parts) - up]) if up else ".".join(parts)
    return f"{base}.{from_module}" if from_module else base


def _search_ast_nodes(
    body: list[ast.stmt], parts: list[str]
) -> tuple[int, int] | None:
    """Recursively search a list of AST statement nodes for a symbol definition.

    Args:
        body: A list of AST nodes, for example ``tree.body`` or
            ``class_node.body``.
        parts: Dotted name split into components, for example
            ``['Class', 'attr']``.

    Returns:
        ``(lineno, end_lineno)`` or ``None``.
    """
    name, *rest = parts
    for node in body:
        if isinstance(
            node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            if node.name == name:
                if rest:
                    return _search_ast_nodes(node.body, rest)
                return node.lineno, node.end_lineno or node.lineno
        elif isinstance(node, (ast.If, ast.Try, ast.With)):
            # Sphinx records attribute docstrings for assignments under
            # these blocks too, so alias lookups must find them here.
            for block in (
                node.body,
                getattr(node, "orelse", []),
                getattr(node, "finalbody", []),
            ):
                result = _search_ast_nodes(block, parts)
                if result:
                    return result
        elif not rest:
            targets: list[ast.expr] = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            for target in targets:
                elements = (
                    target.elts
                    if isinstance(target, (ast.Tuple, ast.List))
                    else [target]
                )
                if any(
                    isinstance(e, ast.Name) and e.id == name for e in elements
                ):
                    return node.lineno, node.end_lineno or node.lineno
    return None


def find_pyi_stub(module_name: str) -> str | None:
    """Find a ``.pyi`` stub file for a C-extension module.

    Derives the expected stub path from the root package's ``__path__``
    (which is reliable for normal Python packages) combined with
    the remaining dotted components of *module_name*.  This avoids depending
    on the C-extension's own ``__path__``, which nanobind sets to the
    directory containing the ``.so`` file — one level too high.

    Falls back to heuristics based on ``__file__`` and ``__path__`` of
    the immediate parent and the module itself when the root-based
    strategy does not find a match.

    Returns:
        Absolute path to the ``.pyi`` file, or ``None``.
    """
    # --- Strategy 1: derive path from root package --------------------
    # For "max._core.driver" the root is "max" (a proper Python package
    # whose __path__ is trustworthy) and the remainder is "_core/driver".
    root_name = module_name.split(".")[0]
    try:
        root_pkg = importlib.import_module(root_name)
    except Exception:
        root_pkg = None

    if root_pkg and "." in module_name:
        rest = module_name[len(root_name) + 1 :]  # e.g. "_core/driver"
        rel_path = rest.replace(".", os.sep)  # e.g. "_core/driver"
        for base in getattr(root_pkg, "__path__", []):
            pyi = os.path.join(base, rel_path + ".pyi")
            if os.path.isfile(pyi):
                return pyi
            pyi = os.path.join(base, rel_path, "__init__.pyi")
            if os.path.isfile(pyi):
                return pyi

    # --- Strategy 2: parent __file__ heuristic ------------------------
    # For single-file extensions (e.g. _core.cpython-313-darwin.so),
    # derive a sibling stub directory from __file__.
    parts = module_name.rsplit(".", 1)
    if len(parts) == 2:
        parent_name, child_name = parts
        try:
            parent = importlib.import_module(parent_name)
        except Exception:
            parent = None

        if parent:
            parent_file = getattr(parent, "__file__", None)
            if parent_file:
                parent_dir = os.path.dirname(parent_file)
                stem = parent_name.rsplit(".", 1)[-1]
                pyi = os.path.join(parent_dir, stem, child_name + ".pyi")
                if os.path.isfile(pyi):
                    return pyi

    # --- Strategy 3: module's own __path__ / __file__ -----------------
    try:
        mod = importlib.import_module(module_name)
    except Exception:
        return None
    for search_path in getattr(mod, "__path__", []):
        pyi = os.path.join(search_path, "__init__.pyi")
        if os.path.isfile(pyi):
            return pyi

    mod_file = getattr(mod, "__file__", None)
    if mod_file:
        mod_dir = os.path.dirname(mod_file)
        stem = module_name.rsplit(".", 1)[-1]
        pyi = os.path.join(mod_dir, stem, "__init__.pyi")
        if os.path.isfile(pyi):
            return pyi

    return None


@functools.cache
def _parse_module(path: str) -> ast.Module | None:
    try:
        with open(path, encoding="utf-8") as f:
            return ast.parse(f.read())
    except (OSError, SyntaxError, ValueError):
        # ValueError covers UnicodeDecodeError and, before Python 3.12,
        # null bytes in the source.
        return None


class Definition(NamedTuple):
    """Where a symbol is defined, as found by :func:`find_definition`."""

    module: str
    name: str
    """Dotted name of the symbol inside the defining module, which differs
    from the requested name for ``from ... import x as y`` re-exports."""
    path: str
    """The ``.py`` source or ``.pyi`` stub holding the definition."""
    lineno: int
    end_lineno: int


@functools.cache
def find_definition(module_name: str, fullname: str) -> Definition | None:
    """Locate where *fullname*, as seen from *module_name*, is defined.

    Follows ``from ... import`` re-exports to the defining module and reads
    the ``.pyi`` stub for C-extension modules.

    Args:
        module_name: Fully-qualified name of the module to start from.
        fullname: Dotted name of the symbol inside that module, for example
            ``'DimLike'`` or ``'Class.attr'``.

    Returns:
        The definition, or ``None`` if no source or stub declares it.
    """
    return _find_definition(module_name, fullname, set())


def _find_definition(
    module_name: str, fullname: str, visited: set[tuple[str, str]]
) -> Definition | None:
    # Re-exports can cycle: ``from . import x`` in a package resolves to
    # the package itself.
    if (module_name, fullname) in visited:
        return None
    visited.add((module_name, fullname))

    try:
        mod = importlib.import_module(module_name)
    except Exception:
        return None

    mod_file = getattr(mod, "__file__", None)
    if not mod_file or not mod_file.endswith((".py", ".pyi")):
        mod_file = find_pyi_stub(module_name)
    if not mod_file:
        return None

    is_package = hasattr(mod, "__path__")
    if mod_file.endswith("__init__.pyi"):
        is_package = True
    parts = fullname.split(".")

    tree = _parse_module(mod_file)
    if tree is None:
        return None

    result = _search_ast_nodes(tree.body, parts)
    if result:
        return Definition(module_name, fullname, mod_file, *result)

    # For re-exports, follow the import chain to the defining module
    top_name = parts[0]
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        for alias in node.names:
            if (alias.asname or alias.name) != top_name:
                continue
            resolved = _resolve_relative_import(
                module_name, node.module, node.level, is_package
            )
            if not resolved:
                continue
            new_fullname = ".".join([alias.name] + parts[1:])
            definition = _find_definition(resolved, new_fullname, visited)
            if definition:
                return definition

    return None
