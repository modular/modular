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

"""Registry and factory for tool-call parsers."""

from __future__ import annotations

from collections.abc import Callable

from max.interfaces import ToolParser

_TOOL_PARSERS: dict[str, type[ToolParser]] = {}


def register(name: str) -> Callable[[type[ToolParser]], type[ToolParser]]:
    """Class decorator that registers a ToolParser under the given name."""

    def decorator(cls: type[ToolParser]) -> type[ToolParser]:
        _TOOL_PARSERS[name] = cls
        return cls

    return decorator


def create(name: str) -> ToolParser:
    """Look up a registered parser by name and instantiate it."""
    cls = _TOOL_PARSERS.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown tool parser: {name!r}. Available: {sorted(_TOOL_PARSERS)}"
        )
    return cls()
