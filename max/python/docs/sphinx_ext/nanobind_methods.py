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
"""Document class-scope nanobind callables as methods.

Nanobind static functions (``nanobind.nb_func``) are not recognized as
methods by the attribute documenter's heuristics, and autodoc's
MethodDocumenter rejects the static methods nanobind attaches to classes
(``Buffer.scalar``, ``Buffer.zeros``, ``Device.cpu``). AttributeDocumenter
would claim them and render ``scalar = <nanobind.nb_func object>`` with
no docstring. Autodoc picks the accepted documenter with the highest
``priority``, and the documenter subclass below outranks
AttributeDocumenter's 10 for nanobind callables. The nanobind
docstring's leading signature line then supplies the signature, as it
does for ``nb_method`` members. Module scope is left untouched
(module-level ``nb_func`` values render through the data documenter
path).

The routine path is guarded by a cross-check against
AttributeDocumenter. Python 3.13 widened ``inspect.isroutine`` to
include every method descriptor (``ismethoddescriptor``), so members
like ``max.graph.Graph.current`` -- a plain descriptor class --
satisfy ``isroutine`` even though they are not callable. Those members
belong to AttributeDocumenter, so this class yields any routine that
the attribute documenter also claims.

Separately, ``sphinx.util.inspect.ismethod`` and ``isclassmethod``
misidentify nanobind bound methods (``nb_method``); a patch here reports
them as plain methods. See
https://github.com/wjakob/nanobind/discussions/707.
"""

from sphinx.application import Sphinx
from sphinx.ext.autodoc import (
    AttributeDocumenter,
    MethodDocumenter,
    ModuleDocumenter,
)
from sphinx.util import inspect as sphinx_inspect


def _is_class_scoped_nanobind_callable(member: object, parent: object) -> bool:
    # ``inspect.isroutine`` is deliberately not consulted here: under
    # Python 3.13 it returns True for nanobind callables too (they are
    # method descriptors), so the check would never fire.
    return (
        not isinstance(parent, ModuleDocumenter)
        and type(member).__module__ == "nanobind"
    )


class NanobindMethodDocumenter(MethodDocumenter):
    """MethodDocumenter that also accepts nanobind static functions."""

    # Must beat AttributeDocumenter (priority 10), or nanobind statics
    # still document as attributes.
    priority = 11

    @classmethod
    def can_document_member(
        cls,
        member: object,
        membername: str,
        isattr: bool,
        parent: object,
    ) -> bool:
        if _is_class_scoped_nanobind_callable(member, parent):
            return True
        if not super().can_document_member(member, membername, isattr, parent):
            return False
        return not AttributeDocumenter.can_document_member(
            member, membername, isattr, parent
        )


def _is_nb_method(obj: object) -> bool:
    return hasattr(obj, "__name__") and type(obj).__name__ == "nb_method"


def _patch_sphinx_inspect() -> None:
    # Sphinx classifies every module member (attribute, function, class,
    # and so on) with the helpers in ``sphinx.util.inspect``, which
    # misidentify nanobind bound methods.
    original_ismethod = sphinx_inspect.ismethod
    original_isclassmethod = sphinx_inspect.isclassmethod

    def _ismethod(obj: object) -> bool:
        if _is_nb_method(obj):
            return True
        return original_ismethod(obj)

    def _isclassmethod(
        obj: object, cls: object = None, name: str | None = None
    ) -> bool:
        if _is_nb_method(obj):
            return False
        return original_isclassmethod(obj, cls, name)

    sphinx_inspect.ismethod = _ismethod
    sphinx_inspect.isclassmethod = _isclassmethod


def setup(app: Sphinx) -> None:
    # Replaces the registered "method" documenter; the automethod
    # directive resolves to the same class. ``override`` also re-registers
    # the directive itself, which skips the "already registered" warning.
    app.add_autodocumenter(NanobindMethodDocumenter, override=True)
    _patch_sphinx_inspect()
