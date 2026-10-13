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
"""Render nanobind signatures and overload docstrings as clean RST.

Nanobind embeds the signature in every ``__doc__`` (one signature line
per C++ overload, followed by one numbered item per overload) and
autodoc does not strip the receiver from extension-method signatures.
The handlers here rewrite all three into what the rendered pages should
say, and the ``setup()`` priorities keep them ahead of napoleon's own
``autodoc-process-docstring`` handler: napoleon is registered when the
extension loads, before conf.py's ``setup()`` runs, and with the default
priority it would treat a leading signature line such as
``set_virtual_device_api(api: str) -> None`` as a "name: description"
header, leaving ``str) -> None`` as the first docstring line. The
overload rewriter must also run before napoleon: it converts the Google
sections itself (they sit below the top level, where napoleon stops) and
napoleon then passes the resulting field lists through untouched.
"""

import inspect
import re
import textwrap
from collections.abc import Callable
from typing import Any

from sphinx.application import Sphinx
from sphinx.ext.autodoc import DocstringSignatureMixin
from sphinx.ext.napoleon import GoogleDocstring

# Matches the same shape as autodoc's ``py_ext_sig_re``: an optional
# module qualifier, the object name, optional type parameters, arguments,
# and an optional return annotation. Kept local so it survives a Sphinx
# version bump changing the original's group layout. Arguments and return
# annotation use balanced parentheses so a prose line that merely starts
# with ``name(...)`` cannot match.
_NB_SIG_RE = re.compile(
    r"""^ ([\w.]+::)?                          # explicit module name
         ([\w.]+\.)?                          # module and/or class name(s)
         (\w+)  \s*                           # thing name
         (?: \[\s*(.*)\s*])?                  # optional: type parameters list
         (?: \(((?:[^()]|\([^()]*\))*)\)      # optional: arguments
          (?:\s* -> \s* ((?:[^()]|\([^()]*\))*))?  #           return annotation
         )? $                                 # and nothing more""",
    re.VERBOSE,
)

OVERLOAD_MARKER = "Overloaded function."

# A leading ``self`` or ``cls`` parameter in a parenthesized argument list.
_RECEIVER_RE = re.compile(r"^\(\s*(?:self|cls)\s*(?:,\s*|(?=\)))")


def _strip_receiver(params: str) -> str:
    """Drop a leading ``self``/``cls`` from ``"(<args>)..."`` text."""
    return _RECEIVER_RE.sub("(", params, count=1)


def strip_signature_lines(
    app: Sphinx,
    what: str,
    name: str,
    obj: Any,
    options: Any,
    lines: list[str],
) -> None:
    """Remove signature lines that nanobind prepends to ``__doc__``.

    Nanobind stores ``"<name>(<args>) -> <ret>"`` as the first line of
    every function and method docstring, followed by the prose docstring.
    ``FunctionDocumenter`` and ``MethodDocumenter`` strip that line via
    ``DocstringSignatureMixin``, but objects documented as data or
    attributes (for example ``nb_func`` values listed in an autosummary
    table, which Sphinx classifies as data) go through
    ``DataDocumenter``, which has no such stripping. Those entries then
    render the raw signature as their description.

    This hook strips leading lines that name the object itself and look
    like a signature, for any documenter. For functions and methods it
    runs before autodoc's own stripping, which then finds no signature
    line to match.
    """
    base = name.rsplit(".", 1)[-1] if name else ""
    if not base:
        return
    valid_names = {base, "__init__"}
    if inspect.isclass(obj):
        valid_names.update(cls.__name__ for cls in getattr(obj, "__mro__", ()))

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line:
            break
        match = _NB_SIG_RE.match(line.strip())
        if not match:
            break
        thing_name = match.group(3)
        if thing_name not in valid_names:
            break
        i += 1

    if i:
        del lines[:i]
        # Drop the blank separator that followed the signature block, so
        # the remaining prose starts at line 0.
        while lines and lines[0].strip() == "":
            del lines[0]


def restructure_overload_docstrings(
    app: Sphinx,
    what: str,
    name: str,
    obj: Any,
    options: Any,
    lines: list[str],
) -> None:
    """Rewrite nanobind's concatenated overload docstring into clean RST.

    For a function with multiple C++ overloads, nanobind builds the
    ``__doc__`` as one signature line per overload, the line "Overloaded
    function.", then one numbered item per overload: ``1. ``<sig>````
    followed by that overload's docstring body. Two problems show up in the
    built pages: each body keeps the C++ source's indentation (docutils
    renders it as a blockquote nested inside the item), and its Google-style
    sections sit below the top level of the docstring, where napoleon only
    converts top-level sections, so ``Args:`` and ``Returns:`` appear
    literally with their indentation artifacts.

    Restructure each item: dedent its body, convert its Google sections with
    napoleon's own parser, and emit it at the top level of the docstring
    under a bold "Overload N" label. Sphinx's ``DocFieldTransformer`` only
    converts field lists at the top level of the object's content, so a body
    nested inside a list item would keep raw ``:param:`` fields. The signature lines before
    the marker are dropped here when ``strip_signature_lines`` has not
    already removed them; each item repeats its own signature anyway.
    """
    if OVERLOAD_MARKER not in lines:
        return
    idx = lines.index(OVERLOAD_MARKER)
    # Only nanobind's layout: everything above the marker is signature
    # lines or blanks (a docstring that merely mentions the marker in
    # prose keeps everything before it and does not match).
    if any(
        line.strip() and not _NB_SIG_RE.match(line.strip())
        for line in lines[:idx]
    ):
        return
    item_re = re.compile(r"^(\d+)\. ``(.*)``\s*$")
    items = []
    current = None
    for line in lines[idx + 1 :]:
        m = item_re.match(line)
        if m:
            current = (m.group(1), m.group(2), [])
            items.append(current)
        elif current is not None:
            current[2].append(line.rstrip())
    if not items:
        return

    def _clean_overload_signature(sig: str) -> str:
        # Nanobind writes each overload's signature with a leading
        # ``self`` and fully-qualified type names. Drop the receiver and
        # the module paths so each entry reads like a Python signature:
        # ``to(devices: Sequence[Device]) -> list[Buffer]``.
        name, paren, rest = sig.partition("(")
        if paren:
            sig = name + _strip_receiver(paren + rest)
        sig = re.sub(r"\b(?:[a-z_][\w]*\.)+([A-Za-z]\w*)", r"\1", sig)
        return sig

    out = [OVERLOAD_MARKER, ""]
    for num, sig, body in items:
        while body and not body[0].strip():
            body.pop(0)
        while body and not body[-1].strip():
            body.pop()
        out.append(f"**Overload {num}:** ``{_clean_overload_signature(sig)}``")
        out.append("")
        if body:
            dedented = textwrap.dedent("\n".join(body)).splitlines()
            converted = GoogleDocstring(dedented, app.config).lines()
            out.extend(line.rstrip() for line in converted)
            out.append("")
    while out and not out[-1]:
        out.pop()
    lines[:] = out


def process_signature(
    app: Sphinx,
    what: str,
    name: str,
    obj: Any,
    options: Any,
    signature: str | None,
    return_annotation: str | None,
) -> tuple[str | None, str | None]:
    """This is a callback from Sphinx that lets us preprocess function signatures.
    We use this to replace "max._core" references with "max"."""
    if signature is not None:
        signature = signature.replace("max._core", "max")

    if return_annotation is not None:
        return_annotation = return_annotation.replace("max._core", "max")

    # Nanobind method signatures carry the receiver (``self``) in the
    # signature text, which autodoc does not strip for extension methods
    # the way it does for Python ones. Drop it so signatures read
    # ``to(device)``.
    if what in ("method", "class") and signature is not None:
        signature = _strip_receiver(signature)

    return signature, return_annotation


def _wrap_find_signature(
    original: Callable[[Any], tuple[str | None, str | None] | None],
) -> Callable[[Any], tuple[str | None, str | None] | None]:
    def _find_signature(self: Any) -> tuple[str | None, str | None] | None:
        # Autodoc reads the second and later docstring signature lines
        # into ``self._signatures`` as ``"(<args>) -> <ret>"`` entries.
        # ``autodoc-process-signature`` only sees the first signature, so
        # apply the same receiver and ``max._core`` rewrites here.
        result = original(self)
        strip_receiver = self.objtype in ("method", "class")
        cleaned = []
        for sig in self._signatures:
            sig = sig.replace("max._core", "max")
            if strip_receiver and sig.startswith("("):
                sig = _strip_receiver(sig)
            cleaned.append(sig)
        self._signatures = cleaned
        return result

    return _find_signature


def setup(app: Sphinx) -> None:
    app.connect("autodoc-process-docstring", strip_signature_lines, priority=1)
    app.connect(
        "autodoc-process-docstring", restructure_overload_docstrings, priority=2
    )
    app.connect("autodoc-process-signature", process_signature)
    DocstringSignatureMixin._find_signature = _wrap_find_signature(
        DocstringSignatureMixin._find_signature
    )
