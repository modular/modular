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

"""Test fixtures for MiniMax M2 tool-call envelopes.

Shared between the grammar tests (PR1) and the parser tests (PR2). Good
envelopes are derived from the model's chat template, vLLM's tool-parser
tests, and the HF model card examples. Bad envelopes are hand-crafted
malformed cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GoodEnvelope:
    name: str
    envelope: str
    expected_calls: list[tuple[str, dict[str, Any]]]


@dataclass(frozen=True)
class BadEnvelope:
    name: str
    envelope: str
    rejection_hint: str


GOOD_ENVELOPES: list[GoodEnvelope] = [
    GoodEnvelope(
        name="01_single_string_arg",
        envelope=(
            "<minimax:tool_call>\n"
            '<invoke name="get_weather">\n'
            '<parameter name="location">San Francisco, CA</parameter>\n'
            "</invoke>\n"
            "</minimax:tool_call>"
        ),
        expected_calls=[("get_weather", {"location": "San Francisco, CA"})],
    ),
    GoodEnvelope(
        name="02_two_string_args",
        envelope=(
            "<minimax:tool_call>\n"
            '<invoke name="get_weather">\n'
            '<parameter name="location">San Francisco, CA</parameter>\n'
            '<parameter name="unit">celsius</parameter>\n'
            "</invoke>\n"
            "</minimax:tool_call>"
        ),
        expected_calls=[
            (
                "get_weather",
                {"location": "San Francisco, CA", "unit": "celsius"},
            )
        ],
    ),
    GoodEnvelope(
        name="03_no_arg_call",
        envelope=(
            "<minimax:tool_call>\n"
            '<invoke name="list_files">\n'
            "</invoke>\n"
            "</minimax:tool_call>"
        ),
        expected_calls=[("list_files", {})],
    ),
    GoodEnvelope(
        name="04_large_args",
        envelope=(
            "<minimax:tool_call>\n"
            '<invoke name="send_email">\n'
            '<parameter name="to">alice@example.com</parameter>\n'
            '<parameter name="subject">Project update for Q2 2026</parameter>\n'
            '<parameter name="body">Hi Alice,\n'
            "\n"
            "I wanted to give you a full rundown of where things stand heading into the end of the quarter.\n"
            "\n"
            "First, the backend migration to the new inference cluster is complete. All traffic has been cut over and we have not seen any regressions in latency or throughput. Peak throughput is up roughly 18 % compared to the previous cluster, which is ahead of the 10 % target we set in January.\n"
            "\n"
            "Second, the new retrieval pipeline shipped last Thursday. The recall@10 numbers on the internal eval set moved from 0.71 to 0.79. We still have a gap versus the external benchmark (0.83) but the team has a clear hypothesis about the re-ranking step and will investigate next sprint.\n"
            "\n"
            "Third, regarding the model bringup work: we have three architectures in active development — MiniMax-M2, a new vision encoder variant, and the distilled 7B checkpoint. All three are tracking green on the nightly smoke tests as of this morning.\n"
            "\n"
            "Finally, budget: we are on track to come in under the compute budget for Q2. I will send the detailed breakdown in a separate spreadsheet before EOD Friday.\n"
            "\n"
            "Let me know if you have questions or want to sync before the all-hands on Monday.\n"
            "\n"
            "Best,\n"
            "James</parameter>\n"
            "</invoke>\n"
            "</minimax:tool_call>"
        ),
        expected_calls=[
            (
                "send_email",
                {
                    "to": "alice@example.com",
                    "subject": "Project update for Q2 2026",
                    "body": (
                        "Hi Alice,\n"
                        "\n"
                        "I wanted to give you a full rundown of where things stand heading into the end of the quarter.\n"
                        "\n"
                        "First, the backend migration to the new inference cluster is complete. All traffic has been cut over and we have not seen any regressions in latency or throughput. Peak throughput is up roughly 18 % compared to the previous cluster, which is ahead of the 10 % target we set in January.\n"
                        "\n"
                        "Second, the new retrieval pipeline shipped last Thursday. The recall@10 numbers on the internal eval set moved from 0.71 to 0.79. We still have a gap versus the external benchmark (0.83) but the team has a clear hypothesis about the re-ranking step and will investigate next sprint.\n"
                        "\n"
                        "Third, regarding the model bringup work: we have three architectures in active development — MiniMax-M2, a new vision encoder variant, and the distilled 7B checkpoint. All three are tracking green on the nightly smoke tests as of this morning.\n"
                        "\n"
                        "Finally, budget: we are on track to come in under the compute budget for Q2. I will send the detailed breakdown in a separate spreadsheet before EOD Friday.\n"
                        "\n"
                        "Let me know if you have questions or want to sync before the all-hands on Monday.\n"
                        "\n"
                        "Best,\n"
                        "James"
                    ),
                },
            )
        ],
    ),
    GoodEnvelope(
        name="05_nested_json_arg",
        envelope=(
            "<minimax:tool_call>\n"
            '<invoke name="update_settings">\n'
            '<parameter name="config">{"theme": "dark", "fontSize": 14}</parameter>\n'
            '<parameter name="enabled">true</parameter>\n'
            "</invoke>\n"
            "</minimax:tool_call>"
        ),
        expected_calls=[
            (
                "update_settings",
                {
                    "config": {"theme": "dark", "fontSize": 14},
                    "enabled": True,
                },
            )
        ],
    ),
    GoodEnvelope(
        name="06_two_invokes_same_function",
        envelope=(
            "<minimax:tool_call>\n"
            '<invoke name="search_web">\n'
            '<parameter name="query_tag">["technology", "events"]</parameter>\n'
            '<parameter name="query_list">["\\"OpenAI\\" \\"latest\\" \\"release\\""]</parameter>\n'
            "</invoke>\n"
            '<invoke name="search_web">\n'
            '<parameter name="query_tag">["technology", "events"]</parameter>\n'
            '<parameter name="query_list">["\\"Gemini\\" \\"latest\\" \\"release\\""]</parameter>\n'
            "</invoke>\n"
            "</minimax:tool_call>"
        ),
        expected_calls=[
            (
                "search_web",
                {
                    "query_tag": ["technology", "events"],
                    "query_list": ['"OpenAI" "latest" "release"'],
                },
            ),
            (
                "search_web",
                {
                    "query_tag": ["technology", "events"],
                    "query_list": ['"Gemini" "latest" "release"'],
                },
            ),
        ],
    ),
    GoodEnvelope(
        name="07_two_invokes_different_functions",
        envelope=(
            "<minimax:tool_call>\n"
            '<invoke name="get_weather">\n'
            '<parameter name="location">NYC</parameter>\n'
            "</invoke>\n"
            '<invoke name="get_stock">\n'
            '<parameter name="ticker">AAPL</parameter>\n'
            "</invoke>\n"
            "</minimax:tool_call>"
        ),
        expected_calls=[
            ("get_weather", {"location": "NYC"}),
            ("get_stock", {"ticker": "AAPL"}),
        ],
    ),
    GoodEnvelope(
        # Renamed from 09_content_after_call: trailing postamble removed when
        # grammar tightened to disallow text after </minimax:tool_call>.
        name="09_eos_after_close",
        envelope=(
            "<minimax:tool_call>\n"
            '<invoke name="web_search">\n'
            '<parameter name="query">latest AI news</parameter>\n'
            "</invoke>\n"
            "</minimax:tool_call>"
        ),
        expected_calls=[("web_search", {"query": "latest AI news"})],
    ),
    GoodEnvelope(
        name="10_string_arg_with_special_chars",
        envelope=(
            "<minimax:tool_call>\n"
            '<invoke name="echo">\n'
            '<parameter name="message">Hello "world" & goodbye</parameter>\n'
            "</invoke>\n"
            "</minimax:tool_call>"
        ),
        expected_calls=[("echo", {"message": 'Hello "world" & goodbye'})],
    ),
]


BAD_ENVELOPES: list[BadEnvelope] = [
    BadEnvelope(
        name="BAD_01_missing_end_tag",
        envelope=(
            "<minimax:tool_call>\n"
            '<invoke name="get_weather">\n'
            '<parameter name="city">NYC</parameter>\n'
            "</invoke>\n"
        ),
        rejection_hint=(
            "No </minimax:tool_call> closing tag — stream never terminates "
            "cleanly; grammar never reaches accepting state."
        ),
    ),
    BadEnvelope(
        name="BAD_02_missing_invoke_end",
        envelope=(
            "<minimax:tool_call>\n"
            '<invoke name="get_weather">\n'
            '<parameter name="city">NYC</parameter>\n'
            "</minimax:tool_call>"
        ),
        rejection_hint=(
            "</invoke> is absent; </minimax:tool_call> appears while still "
            "inside <invoke> — grammar rejects."
        ),
    ),
    BadEnvelope(
        name="BAD_03_unknown_tag_inside",
        envelope=(
            "<minimax:tool_call>\n"
            '<invoke name="get_weather">\n'
            '<parameter name="city">NYC</parameter>\n'
            "        <unknown_tag>bad</unknown_tag>\n"
            "</invoke>\n"
            "</minimax:tool_call>"
        ),
        rejection_hint=(
            "<unknown_tag> is not a <parameter> — grammar rejects any tag "
            'other than <parameter name="..."> inside <invoke>.'
        ),
    ),
    BadEnvelope(
        name="BAD_04_garbage_prefix",
        envelope=(
            "GARBAGE PREFIX<minimax:tool_call>\n"
            '<invoke name="get_weather">\n'
            '<parameter name="city">NYC</parameter>\n'
            "</invoke>\n"
            "</minimax:tool_call>"
        ),
        rejection_hint=(
            "Grammar starts at <minimax:tool_call>; any byte before that "
            "(here 'G' in 'GARBAGE PREFIX') is rejected at step 0."
        ),
    ),
    BadEnvelope(
        name="BAD_05_malformed_attribute",
        envelope=(
            "<minimax:tool_call>\n"
            "<invoke name=>\n"
            '<parameter name="city">NYC</parameter>\n'
            "</invoke>\n"
            "</minimax:tool_call>"
        ),
        rejection_hint=(
            'name=> has empty attribute value; grammar requires name="[^"]+"> '
            "(at least one character) — rejects."
        ),
    ),
]
