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
"""Sphinx build patches for the MAX Python API reference.

The modules in this package patch the Sphinx markdown pipeline where
``max/python/docs/conf.py.in`` cannot express the behavior as plain
configuration:

- :mod:`signatures` handles nanobind's embedded signature lines and
  concatenated overload docstrings.
- :mod:`docstring_sources` resolves docstrings through re-export chains
  and ``.pyi`` stubs when the runtime object carries none.
- :mod:`nanobind_methods` teaches autodoc that a class-scope nanobind
  callable is a method.
- :mod:`typing_xrefs` cross-references ``typing`` names against the
  Python inventory and resolves type-alias references.
- :mod:`markdown_builder` repairs two table and signature renderings in
  sphinx_markdown_builder.
- :mod:`source_files` locates source files, stubs, and symbols; it backs
  the modules above and the ``linkcode_resolve`` entry point in conf.py.
"""

from sphinx.application import Sphinx

from sphinx_ext import (
    docstring_sources,
    markdown_builder,
    nanobind_methods,
    signatures,
    typing_xrefs,
)


def setup(app: Sphinx) -> None:
    """Register every patch with the Sphinx application."""
    signatures.setup(app)
    docstring_sources.setup(app)
    nanobind_methods.setup(app)
    typing_xrefs.setup(app)
    markdown_builder.setup(app)
