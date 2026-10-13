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
"""Repair sphinx_markdown_builder renderings that GFM cannot express.

Two patches: escape literal pipes in table cells (GFM's column
separator), and drop the id-less duplicate overload signature nodes that
render as stray blockquotes.
"""

from docutils import nodes
from sphinx.application import Sphinx

# Imported unconditionally: conf.py requires sphinx_markdown_builder, so a
# missing module means an upgrade moved it, and the build fails loudly.
from sphinx_markdown_builder.contexts import TableContext
from sphinx_markdown_builder.translator import MarkdownTranslator

from sphinx_ext.signatures import OVERLOAD_MARKER


def setup(app: Sphinx) -> None:
    # GFM tables use "|" as the column separator, so a literal "|" inside
    # a cell must be escaped. sphinx_markdown_builder does not escape
    # them, which breaks a table whenever a summary or signature contains
    # a pipe (for example a union type such as "int | str"). Escape the
    # pipe in every rendered cell, the way the GFM spec prescribes.
    _original_make_row = TableContext.make_row

    def _make_row_escaping_pipes(row: list[str]) -> list[str]:
        escaped = _original_make_row(row)
        return [cell.replace("|", "\\|") for cell in escaped]

    TableContext.make_row = staticmethod(_make_row_escaping_pipes)

    # For a function with C++ overloads, autodoc emits one desc_signature
    # node per overload: the first carries the anchor id, the rest carry
    # none. sphinx_markdown_builder renders the id-less ones as markdown
    # blockquotes ("> sig"), which reads as a stray quotation line above
    # the restructured "Overloaded function." entries -- and each numbered
    # entry already repeats its own signature. Skip the id-less nodes, but
    # only when the description's content is the nanobind overload layout,
    # so a genuinely distinct signature is never hidden.
    _original_visit_desc_signature = MarkdownTranslator.visit_desc_signature

    def _visit_desc_signature_skip_overloads(
        self: MarkdownTranslator, node: nodes.Element
    ) -> None:
        if (
            not self.status.cli_desc_option
            and not self.status.cli_desc_object
            and not node.get("ids")
            and node.parent is not None
        ):
            siblings = [
                s for s in node.parent.children if s.tagname == "desc_signature"
            ]
            # Only the description's own top-level paragraphs: a class's
            # content also holds its methods, and an overloaded method
            # must not hide the class's own constructor overloads.
            content = next(
                (
                    c
                    for c in node.parent.children
                    if c.tagname == "desc_content"
                ),
                None,
            )
            if (
                len(siblings) > 1
                and content is not None
                and any(
                    isinstance(c, nodes.paragraph)
                    and c.astext() == OVERLOAD_MARKER
                    for c in content.children
                )
            ):
                raise nodes.SkipNode()
        return _original_visit_desc_signature(self, node)

    MarkdownTranslator.visit_desc_signature = (
        _visit_desc_signature_skip_overloads
    )
