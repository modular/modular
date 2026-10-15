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
"""Find real docstrings for objects autodoc cannot analyze.

Nanobind objects, re-exported aliases, and constants carrying inherited
builtin docstrings all reach autodoc with a missing or wrong docstring.
The patches here give the documenters two extra sources: the attribute
docstring recorded next to the assignment in the defining module (found
by walking ``from ... import`` re-export chains), and the hand-maintained
``.pyi`` stub for C-extension objects. A napoleon patch keeps prose
summaries of module constants from being mangled into Google's inline
attribute form.
"""

import ast
import functools
import inspect
import logging
import re
from collections.abc import Callable

from sphinx.application import Sphinx
from sphinx.errors import PycodeError
from sphinx.ext.autodoc import (
    AttributeDocumenter,
    ClassDocumenter,
    DataDocumenter,
    Documenter,
)
from sphinx.ext.napoleon import GoogleDocstring
from sphinx.pycode import ModuleAnalyzer
from sphinx.util.docstrings import prepare_docstring

from sphinx_ext.source_files import find_definition, find_pyi_stub

logger = logging.getLogger(__name__)

_stub_doc_cache = {}


def _attr_doc_from_definition(modname: str, attrname: str) -> list[str] | None:
    """Return the attribute docstring of *attrname* from its defining
    module, or ``None`` when *modname* defines it itself.
    """
    definition = find_definition(modname, attrname)
    if definition is None or definition.module == modname:
        return None
    try:
        analyzer = ModuleAnalyzer.for_module(definition.module)
        analyzer.analyze()
    except PycodeError:
        return None
    key = ("", definition.name)
    if key in analyzer.attr_docs:
        return list(analyzer.attr_docs[key])
    return None


def _stub_docstring(module_name: str, qualname: str) -> str | None:
    """Return the docstring recorded for *qualname* in the module's
    ``.pyi`` stub, or ``None`` when the stub does not document it.
    """
    cache_key = (module_name, qualname)
    if cache_key in _stub_doc_cache:
        return _stub_doc_cache[cache_key]

    result = None
    stub_path = find_pyi_stub(module_name) if module_name else None
    if stub_path:
        try:
            with open(stub_path, encoding="utf-8") as f:
                tree = ast.parse(f.read())
            node = None
            parts = qualname.split(".")
            for candidate in tree.body:
                if getattr(candidate, "name", None) == parts[0]:
                    node = candidate
                    break
            for part in parts[1:]:
                if node is None:
                    break
                node = next(
                    (
                        child
                        for child in getattr(node, "body", [])
                        if getattr(child, "name", None) == part
                    ),
                    None,
                )
            if node is not None:
                result = ast.get_docstring(node)
        except Exception:
            logger.debug(
                "failed to parse .pyi stub for %s: %s",
                module_name,
                stub_path,
                exc_info=True,
            )
    _stub_doc_cache[cache_key] = result
    return result


def _inherits_type_docstring(obj: object, lines: list[str]) -> bool:
    """Report whether *lines* hold the built-in docstring of *obj*'s type.

    ``Documenter.get_doc`` resolves the docstring through the runtime
    object, so an undocumented module constant such as
    ``ADAPTER_CONFIG_FILE = "adapter_config.json"`` reports ``str.__doc__``
    through the instance and renders pages of builtin boilerplate.
    ``type[...]`` aliases leak the same way, delegating attribute lookup
    to the origin class. Treat those as undocumented so the stub fallback
    (and the attribute-comment machinery, which runs before this) can
    supply the real description. An undocumented constant renders with an
    empty description.
    """
    if (
        obj is None
        or inspect.isclass(obj)
        or inspect.ismodule(obj)
        or inspect.isroutine(obj)
    ):
        return False
    text = "\n".join(lines).strip()
    if not text:
        return False
    candidates = []
    try:
        candidates.append(type(obj).__doc__)
    except Exception:
        pass
    origin = getattr(obj, "__origin__", None)
    if origin is not None:
        candidates.append(getattr(origin, "__doc__", None))
    # ``__doc__`` can be a getset_descriptor (unbound attribute access
    # on the type); only string candidates can ever match.
    candidates = [c for c in candidates if isinstance(c, str)]
    return any(c.strip() == text for c in candidates if c)


def _make_get_doc_with_stub_fallback(
    original_get_doc: Callable[[Documenter], list[list[str]] | None],
) -> Callable[[Documenter], list[list[str]] | None]:
    def _get_doc_with_stub_fallback(
        self: Documenter,
    ) -> list[list[str]] | None:
        docstrings = original_get_doc(self)
        if docstrings and _inherits_type_docstring(self.object, docstrings[0]):
            docstrings = []
        if docstrings:
            return docstrings
        # ClassDocumenter documents module-level type aliases
        # (``DimLike``, ``GPUProfilingMode``) with ``autoclass``; for a
        # re-exported alias it returns ``None`` without ever consulting
        # the defining module, so the alias page renders a bare
        # ``alias of ...`` line with no description. Supply the
        # attribute docstring recorded next to the assignment in the
        # defining module.
        if getattr(self, "doc_as_attr", False) and self.objpath:
            attr_doc = _attr_doc_from_definition(self.modname, self.objpath[-1])
            if attr_doc:
                return [attr_doc]
        obj_module = getattr(self.object, "__module__", None)
        if not obj_module or not obj_module.startswith("max."):
            return docstrings
        qualname = ".".join(self.objpath)
        if not qualname:
            return docstrings
        stub_doc = _stub_docstring(obj_module, qualname)
        if not stub_doc:
            return docstrings
        tab_width = self.directive.state.document.settings.tab_width
        return [prepare_docstring(stub_doc, tab_width)]

    return _get_doc_with_stub_fallback


# Napoleon parses every ``data``/``attribute`` docstring as Google's
# inline attribute form and splits the first line at the first colon,
# filing the text before the colon under ``:type:``. Module-level
# aliases and constants carry prose summaries instead
# ("Anything a dimension accepts: a Python int, ..."), so napoleon
# eats the lead-in and leaves the tail as the description, starting
# mid-sentence. Keep the split only when the text before the colon is
# an identifier, optionally with a parenthesized type -- the
# documented Google attribute form ("name (type): description").
_INLINE_ATTR_NAME_RE = re.compile(r"^\s*[\w.]+\s*(\([^()]*\))?\s*$")
# Constants that open prose ("False: a state's rows hold ...") match the
# identifier pattern but are never a type.
_INLINE_ATTR_PROSE_WORDS = frozenset({"True", "False", "None"})


def _is_prose_lead_in(text: str) -> bool:
    return (
        not _INLINE_ATTR_NAME_RE.match(text)
        or text.strip() in _INLINE_ATTR_PROSE_WORDS
    )


def _wrap_consume_inline_attribute(
    original: Callable[[GoogleDocstring], tuple[str, list[str]]],
) -> Callable[[GoogleDocstring], tuple[str, list[str]]]:
    def _consume_inline_attribute(
        self: GoogleDocstring,
    ) -> tuple[str, list[str]]:
        _type, descs = original(self)
        if _type and descs and _is_prose_lead_in(_type):
            # A prose lead-in, not the attribute form: rejoin the line
            # as the description and drop the bogus ``:type:``. Napoleon
            # strips the space after the colon when it splits the line.
            return "", [f"{_type}: {descs[0]}", *descs[1:]]
        return _type, descs

    return _consume_inline_attribute


@functools.cache
def _pyi_analyzer(module_name: str) -> ModuleAnalyzer | None:
    """Return a ``ModuleAnalyzer`` built from the ``.pyi`` stub for
    *module_name*, or ``None`` if no stub exists.
    """
    pyi_path = find_pyi_stub(module_name)
    if not pyi_path:
        return None
    try:
        with open(pyi_path, encoding="utf-8") as f:
            source = f.read()
        analyzer = ModuleAnalyzer.for_string(source, module_name, pyi_path)
        analyzer.analyze()
    except (OSError, UnicodeDecodeError, PycodeError):
        logger.debug(
            "failed to parse .pyi stub for %s: %s",
            module_name,
            pyi_path,
            exc_info=True,
        )
        return None
    return analyzer


def _make_get_attribute_comment_with_pyi(
    original: Callable[[AttributeDocumenter, object, str], list[str] | None],
) -> Callable[[AttributeDocumenter, object, str], list[str] | None]:
    def _get_attribute_comment_with_pyi(
        self: AttributeDocumenter, parent: object, attrname: str
    ) -> list[str] | None:
        """Extend ``get_attribute_comment`` with a ``.pyi`` fallback.

        Tries the original method first.  If it returns ``None`` (e.g.
        because the module is a C extension with no ``.py`` source),
        walks the MRO and looks up the attribute docstring in the
        ``.pyi`` stub's parsed ``attr_docs``.
        """
        result = original(self, parent, attrname)
        if result is not None:
            return result
        if not inspect.isclass(parent):
            return None
        for cls in inspect.getmro(parent):
            module_name = getattr(cls, "__module__", None)
            qualname = getattr(cls, "__qualname__", None)
            if not module_name or not qualname:
                continue
            analyzer = _pyi_analyzer(module_name)
            if analyzer is None:
                continue
            key = (qualname, attrname)
            if key in analyzer.attr_docs:
                return list(analyzer.attr_docs[key])
        return None

    return _get_attribute_comment_with_pyi


def setup(app: Sphinx) -> None:
    # DataDocumenter.get_module_comment reports the attribute docstring
    # recorded next to the assignment in the documented module only.
    # Pages document aliases like ``max.engine.GPUProfilingMode`` through
    # the package that re-exports them, so the docstring in the defining
    # module is invisible to the analyzer of the re-exporting package.
    # Fall back to walking the ``from ... import`` chain until the
    # defining module is found and read the docstring there.
    _original_get_module_comment = DataDocumenter.get_module_comment

    def _get_module_comment_with_reexports(
        self: DataDocumenter, attrname: str
    ) -> list[str] | None:
        result = _original_get_module_comment(self, attrname)
        if result is not None:
            return result
        return _attr_doc_from_definition(self.modname, attrname)

    DataDocumenter.get_module_comment = _get_module_comment_with_reexports

    # Fall back to .pyi stub docstrings when the runtime object has none.
    # Nanobind exceptions registered with ``nb::exception`` cannot carry a
    # docstring from C++, and the hand-maintained stub is the only place
    # documenting them. Parsing the stub for the object's own docstring
    # makes the stub the source of truth for such objects.
    Documenter.get_doc = _make_get_doc_with_stub_fallback(Documenter.get_doc)
    # ClassDocumenter defines its own ``get_doc`` that never calls the base
    # class implementation, so the fallback above never runs for classes.
    # ExceptionDocumenter (nanobind exceptions such as HostHazardError)
    # inherits from it, so patch it the same way.
    ClassDocumenter.get_doc = _make_get_doc_with_stub_fallback(
        ClassDocumenter.get_doc
    )

    # Fall back to .pyi stubs for attribute docstrings when the .py
    # source is unavailable (C extensions). The stub analyzer is scoped to
    # attribute-docstring lookup only; feeding .pyi stubs into the global
    # ModuleAnalyzer cache would break Sphinx's handling of @overload and
    # other stub-only constructs.
    AttributeDocumenter.get_attribute_comment = (
        _make_get_attribute_comment_with_pyi(
            AttributeDocumenter.get_attribute_comment
        )
    )

    # Wraps a private napoleon method; reading it raises AttributeError if
    # a napoleon upgrade renames it.
    GoogleDocstring._consume_inline_attribute = _wrap_consume_inline_attribute(
        GoogleDocstring._consume_inline_attribute
    )
