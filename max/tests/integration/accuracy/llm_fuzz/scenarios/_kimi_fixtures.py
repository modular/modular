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
"""Shared Kimi K2.5 fuzz fixtures.

Schemas and payload shapes that are referenced from more than one
scenario (so the production-observed input lives in one place; if the
captured shape ever changes, both scenarios update together).
"""

from __future__ import annotations

from typing import Any

# Production-observed tool schema. Verbatim from the engine log line
# that preceded every TypeScript-conversion warning during the
# 2026-05-23 Kimi K2.5 freeze, modulo wrapping in a function-tool
# envelope and giving it a name. The ``position`` field's ``oneOf``
# whose first branch is a bare ``{"const": "end"}`` literal is the
# construct Kimi's bundled HF tokenizer
# (``tool_declaration_ts.py:_parse_parameter_type``) refuses to
# recognise, which is what dropped the tool declarations in
# production.
PRODUCTION_ONEOF_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "place_item",
        "description": "Place an item at a position in a list.",
        "parameters": {
            "type": "object",
            "properties": {
                "item": {
                    "type": "string",
                    "description": "Name of the item to place.",
                },
                "position": {
                    "description": (
                        "Where to insert (default: end). Only for outline "
                        "items."
                    ),
                    "oneOf": [
                        {"const": "end"},
                        {
                            "type": "object",
                            "properties": {"after": {"type": "string"}},
                            "required": ["after"],
                        },
                        {
                            "type": "object",
                            "properties": {
                                "index": {
                                    "type": "integer",
                                    "minimum": 0,
                                }
                            },
                            "required": ["index"],
                        },
                    ],
                },
            },
            "required": ["item", "position"],
        },
    },
}
