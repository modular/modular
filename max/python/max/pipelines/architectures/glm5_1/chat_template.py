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
# The Jinja excerpts in this file (_QUADRATIC_TOOL_RESULT_BLOCK and
# _IN_ORDER_TOOL_RESULT_BLOCK) are copied from GLM-5.3's chat_template.jinja
# and are distributed under the GLM-5.3 License:
#
# Copyright (c) 2026 Z.AI
#
# Permission is hereby granted, free of charge, to any person or entity (the
# "Licensee") obtaining a copy of this software — including the model weights,
# parameters, configuration files, inference and training code, and associated
# documentation (collectively, the "Software") — to deal in the Software
# without restriction, including without limitation the rights to use, copy,
# modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software; to run, deploy, fine-tune, or otherwise modify the Software and
# create derivative works from it; and to permit persons to whom the Software
# is furnished to do so, subject to the following conditions:
#
# 1. The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software. The Licensee's use of
# the Software must comply with applicable laws and regulations.
#
# 2. "Model as a Service" means giving a third party access to language model
# inference or fine-tuning (e.g., via API) in a manner that allows such third
# party to exercise meaningful control over the inputs, parameters, or
# training data. This does not include (a) end-user products with model
# capabilities solely embedded within specific features or harnesses, or (b)
# mere relaying of requests to models hosted by others.
# If the Licensee or any of its affiliates operates a Model as a Service
# business, and the aggregate revenue of the Licensee and its affiliates
# exceeds 10 billion US dollars (or the equivalent in other currencies) in
# total over any consecutive 12 months, the Licensee must pass Z.AI's security
# review before using the Software or its derivative works for any commercial
# purpose. The scope and method of the security review shall be reasonably
# determined by Z.AI.
#
# 3. THE SOFTWARE AND ANY OUTPUT AND RESULTS THEREFROM ARE PROVIDED ON AN "AS
# IS" BASIS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL Z.AI OR ITS AFFILIATES OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
# IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# For any questions regarding this license, please contact glmlicense@z.ai.

"""Linear-time rendering of tool results for the GLM-5.3 chat template.

GLM-5.3's template renders a run of ``tool`` messages in the order of the
preceding assistant turn's ``tool_calls`` when every result id is present,
unique, and matches a call. It decides this with nested loops of Jinja macros,
which costs O(n^2) in the number of tool calls in one turn: 1,000 calls take
about 17 s to render and 2,000 about 70 s. A history with thousands of calls
in one turn therefore stalls the server before the prompt-length check can
reject it.

The fix splits that decision out of the template. :func:`linearize_tool_results`
replaces the quadratic block with the template's own in-order fallback loop,
and :func:`order_tool_results` reorders the messages in Python beforehand,
applying the same rules in O(n). The two together render the same prompt as
the unmodified template.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from max.pipelines.modeling.types import TextGenerationRequestMessage

# Verbatim from GLM-5.3's chat_template.jinja (zai-org/GLM-5.3 and
# RadixArk/GLM-5.3-NVFP4 ship the same file). Matched exactly, so a template
# that differs here in any way is left alone.
_QUADRATIC_TOOL_RESULT_BLOCK = """\
    {%- set ns_chk = namespace(can_sort=true) -%}
    {%- if not ns_a.tool_calls -%}
        {%- set ns_chk.can_sort = false -%}
    {%- else -%}
        {%- for k in range(block_start, ns_blk.end + 1) -%}
            {%- if not ns_chk.can_sort -%}{%- break -%}{%- endif -%}
            {%- set m = messages[k] -%}
            {%- if is_list_of_outputs(m) -%}
                {%- for entry in m.content -%}
                    {%- if not ns_chk.can_sort -%}{%- break -%}{%- endif -%}
                    {%- set eid = id_of(entry) -%}
                    {%- if not eid -%}
                        {%- set ns_chk.can_sort = false -%}
                    {%- elif has_dup_tool_result_id(block_start, ns_blk.end, eid) -%}
                        {%- set ns_chk.can_sort = false -%}
                    {%- elif not tc_id_exists(ns_a.tool_calls, eid) -%}
                        {%- set ns_chk.can_sort = false -%}
                    {%- endif -%}
                {%- endfor -%}
            {%- else -%}
                {%- set tk_id = id_of(m) -%}
                {%- if not tk_id -%}
                    {%- set ns_chk.can_sort = false -%}
                {%- elif has_dup_tool_result_id(block_start, ns_blk.end, tk_id) -%}
                    {%- set ns_chk.can_sort = false -%}
                {%- elif not tc_id_exists(ns_a.tool_calls, tk_id) -%}
                    {%- set ns_chk.can_sort = false -%}
                {%- endif -%}
            {%- endif -%}
        {%- endfor -%}
        {%- for i in range(ns_a.tool_calls | length) -%}
            {%- if not ns_chk.can_sort -%}{%- break -%}{%- endif -%}
            {%- set tc_id = id_of(ns_a.tool_calls[i]) -%}
            {%- if not tc_id -%}
                {%- set ns_chk.can_sort = false -%}
            {%- endif -%}
            {%- for j in range(i + 1, ns_a.tool_calls | length) -%}
                {%- if id_of(ns_a.tool_calls[j]) == tc_id -%}
                    {%- set ns_chk.can_sort = false -%}
                    {%- break -%}
                {%- endif -%}
            {%- endfor -%}
        {%- endfor -%}
    {%- endif -%}
    {%- if ns_chk.can_sort -%}
        {%- for tc in ns_a.tool_calls -%}
            {%- set tc_id = id_of(tc) -%}
            {%- for k in range(block_start, ns_blk.end + 1) -%}
                {%- set m = messages[k] -%}
                {%- if is_list_of_outputs(m) -%}
                    {%- for entry in m.content -%}
                        {%- set eid = id_of(entry) -%}
                        {%- if eid == tc_id -%}
                            {%- if entry.output is iterable and entry.output is not string and entry.output is not mapping and entry.output and entry.output.0.type == "tool_reference" -%}
                                {{- tool_references_to_response(entry.output) -}}
                            {%- else -%}
                                {{- tool_response(visible_text(entry.output)) -}}
                            {%- endif -%}
                        {%- endif -%}
                    {%- endfor -%}
                {%- else -%}
                    {%- set tk_id = id_of(m) -%}
                    {%- if tk_id == tc_id -%}
                        {{- render_tool_response(m) -}}
                    {%- endif -%}
                {%- endif -%}
            {%- endfor -%}
        {%- endfor -%}
    {%- else -%}
        {%- for k in range(block_start, ns_blk.end + 1) -%}
            {{- render_tool_response(messages[k]) -}}
        {%- endfor -%}
    {%- endif -%}"""

# The template's own fallback branch: render the block's results in order.
_IN_ORDER_TOOL_RESULT_BLOCK = """\
        {%- for k in range(block_start, ns_blk.end + 1) -%}
            {{- render_tool_response(messages[k]) -}}
        {%- endfor -%}"""


def linearize_tool_results(template: str) -> str | None:
    """Replaces GLM-5.3's quadratic tool-result ordering with an in-order loop.

    The returned template no longer reorders tool results itself, so it must
    be rendered with messages that went through :func:`order_tool_results`.

    Args:
        template: The chat template source.

    Returns:
        The patched template, or ``None`` if ``template`` does not contain
        GLM-5.3's tool-result block exactly once.
    """
    if template.count(_QUADRATIC_TOOL_RESULT_BLOCK) != 1:
        return None
    return template.replace(
        _QUADRATIC_TOOL_RESULT_BLOCK, _IN_ORDER_TOOL_RESULT_BLOCK
    )


def _tool_call_id(call: object) -> str:
    """Mirrors the template's ``id_of`` macro for one entry of ``tool_calls``."""
    if not isinstance(call, Mapping):
        return ""
    for key in ("tool_call_id", "id"):
        value = call.get(key)
        if value:
            return str(value)
    return ""


def _sorted_block(
    results: Sequence[TextGenerationRequestMessage],
    tool_calls: Sequence[object],
) -> list[TextGenerationRequestMessage] | None:
    """Orders one run of tool results by the calls they answer.

    Returns ``None`` where the template would keep the received order: a
    result or call without an id, a duplicated id, or a result whose id
    matches no call.
    """
    position: dict[str, int] = {}
    for index, call in enumerate(tool_calls):
        call_id = _tool_call_id(call)
        if not call_id or call_id in position:
            return None
        position[call_id] = index

    keyed: list[tuple[int, TextGenerationRequestMessage]] = []
    seen: set[str] = set()
    for result in results:
        result_id = str(result.tool_call_id) if result.tool_call_id else ""
        if not result_id or result_id in seen or result_id not in position:
            return None
        seen.add(result_id)
        keyed.append((position[result_id], result))
    keyed.sort(key=lambda pair: pair[0])
    return [result for _, result in keyed]


def order_tool_results(
    messages: Sequence[TextGenerationRequestMessage],
) -> list[TextGenerationRequestMessage]:
    """Applies GLM-5.3's tool-result ordering in Python, in O(n).

    Each maximal run of ``tool`` messages that directly follows an assistant
    turn with ``tool_calls`` is sorted into the order of those calls, under the
    same conditions as the template (see :func:`_sorted_block`). Every other
    message keeps its position.

    The template also accepts results whose content is a list of
    ``{output: ...}`` entries. Flattened request messages always carry string
    content, so that form never reaches it from MAX and is not handled here.

    Args:
        messages: The conversation, as passed to the chat template.

    Returns:
        A new list with each tool-result run in the order the template would
        render it.
    """
    ordered: list[TextGenerationRequestMessage] = []
    i = 0
    while i < len(messages):
        if str(messages[i].role) != "tool":
            ordered.append(messages[i])
            i += 1
            continue
        end = i
        while end < len(messages) and str(messages[end].role) == "tool":
            end += 1
        block = messages[i:end]
        previous = messages[i - 1] if i > 0 else None
        sorted_block = None
        if (
            previous is not None
            and str(previous.role) == "assistant"
            and previous.tool_calls
        ):
            sorted_block = _sorted_block(block, previous.tool_calls)
        ordered.extend(sorted_block if sorted_block is not None else block)
        i = end
    return ordered
