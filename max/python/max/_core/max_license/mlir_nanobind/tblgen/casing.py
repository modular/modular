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
"""Casing helpers for backends."""

import re


def camel_to_snake_case(name: str) -> str:
    snake_name = ""
    start = 0
    for match in re.finditer(r"([A-Z0-9]+)([^A-Z]*)", name):
        snake_name += name[start : match.start()]
        if match.group(2):
            if len(match.group(1)) > 1:
                upper_word = match.group(1)[:-1]
                snake_name += "_" + upper_word.lower()
            next_word = match.group(1)[-1] + match.group(2)
            snake_name += "_" + next_word.lower()
        else:
            snake_name += "_" + match.group(0).lower()
        start = match.end()
    if start < len(name):
        if snake_name:
            snake_name += "_"
        snake_name += name[start:].lower()
    return snake_name


def snake_to_camel_case(name: str) -> str:
    camel_name = ""
    if name.startswith("_"):
        camel_name = "_"
        name = name[1:]
    start = 0
    for match in re.finditer(r"_(\w)", name):
        camel_name += name[start : match.start()]
        camel_name += match.group(1).upper()
        start = match.end()
    camel_name += name[start:]
    return camel_name


def snake_to_upper_camel_case(name: str) -> str:
    camel_name = snake_to_camel_case(name)
    first_letter = re.search(r"\w", camel_name)
    if first_letter:
        camel_name = (
            camel_name[: first_letter.start()]
            + camel_name[first_letter.start()].upper()
            + camel_name[first_letter.start() + 1 :]
        )
    return camel_name


def camel_to_upper_camel(name: str) -> str:
    c, rest = name[0], name[1:]
    return c.upper() + rest


def camel_to_lower_camel(name: str) -> str:
    c, rest = name[0], name[1:]
    return c.lower() + rest


def is_snake_case(name: str):  # noqa: ANN201
    return "_" in name or name.islower()


def is_camel_case(name: str):  # noqa: ANN201
    return "_" not in name


def camel_case(name: str) -> str:
    if is_camel_case(name):
        return camel_to_lower_camel(name)
    return snake_to_camel_case(name)


def upper_camel_case(name: str) -> str:
    return camel_to_upper_camel(camel_case(name))


def snake_case(name: str) -> str:
    return name if is_snake_case(name) else camel_to_snake_case(name)
