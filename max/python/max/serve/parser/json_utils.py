# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

import json
from typing import Any

import regex

_JSON_REGEX_STR = r"""
(?(DEFINE)
    (?<ws>      [\t\n\r ]* )
    (?<number>  -? (?: 0|[1-9]\d*) (?: \.\d+)? (?: [Ee] [+-]? \d++)? )
    (?<boolean> true | false | null )
    (?<string>  " (?: [^\\\\"\x00-\x1f] | \\\\ ["\\\\bfnrt\/] | \\\\u[0-9A-Fa-f]{4} )* " )
    (?<pair>    (?&ws)(?&string)(?&ws):(?&value) )
    (?<array>   \[ (?&ws) (?: (?&value) (?&ws) (?: , (?&ws) (?&value) (?&ws) )* )? \] )
    (?<object>  \{ (?&ws) (?: (?&pair) (?&ws) (?: , (?&ws) (?&pair) (?&ws) )* )? \} )
    (?<value>   (?&ws)(?: (?&number) | (?&boolean) | (?&string) | (?&array) | (?&object) )(?&ws) )
)
(?P<json>(?&value))
"""


_JSON_REGEX = regex.compile(_JSON_REGEX_STR, regex.VERBOSE | regex.DOTALL)  # type: ignore[attr-defined]


def parse_json_from_text(text: str) -> list[Any]:
    """Parse JSON from text."""
    json_objects: list[Any] = []
    cursor = 0
    while cursor < len(text):
        match = _JSON_REGEX.search(text, pos=cursor)
        if not match:
            break
        json_val = match.group("json")
        json_objects.append(json.loads(json_val))
        cursor = match.end()

    return json_objects
