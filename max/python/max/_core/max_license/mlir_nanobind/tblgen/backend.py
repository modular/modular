# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the MAX License:
# https://modular.com/legal/max-license
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Base class and helpers for backends."""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, TypeVar

from .types import Entry

T = TypeVar("T")
T_contra = TypeVar("T_contra", contravariant=True)


def indent(text: str, n: int = 0, indent_str: str = "    ") -> str:
    text = text.strip("\n")
    if match := re.match(r"^ *", text):
        text = re.sub(
            f"^ {{{len(match.group(0))}}}", "", text, flags=re.MULTILINE
        )
    return re.sub("^", indent_str * n, text, flags=re.MULTILINE)


def indent_all(texts: Iterable[str], n: int = 0, indent_str: str = "    "):  # noqa: ANN201
    for text in texts:
        yield indent(text, n, indent_str)


@dataclass
class BackendTypeRegistry:
    generators: dict = dataclasses.field(default_factory=dict)  # type: ignore[type-arg]

    # mypy can't actually express that T_contra should be bound by T, but it's close enough
    def generator(self, type: type[T]):  # noqa: ANN201
        def decorator(fn: Callable[[Any, T_contra], Iterable[str]]):  # noqa: ANN202
            self.generators[type] = fn
            return fn

        return decorator


class Backend:
    registry: BackendTypeRegistry

    def generate_all(self, entries: Iterable[Entry]):  # noqa: ANN201
        for entry in entries:
            yield from self.generate(entry)

    def generate(self, entry: Entry):  # noqa: ANN201
        for superclass in type(entry).__mro__:
            if generator := self.registry.generators.get(superclass):
                yield from generator(self, entry)
                return
