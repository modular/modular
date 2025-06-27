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

from collections.abc import Iterable
from pathlib import Path, PosixPath, WindowsPath
from typing import (
    IO,
    Any,
    BinaryIO,
    Optional,
    TypeVar,
    Union,
)

import rich
from ruamel import yaml
from ruamel.yaml.compat import StringIO


def pretty_exception_handler(exception_type, exception, traceback):
    rich.print(f"[bold red]{exception_type.__name__}[/bold red]: {exception}")


StreamType = Union[BinaryIO, IO[str], StringIO]


# TODO: replace the calls to YAML with direct calls to ruamel
class NoAliasRepresenter(yaml.RoundTripRepresenter):
    """A custom YAML representer that ignores aliases.

    Aliases are defined by the YAML spec
    https://yaml.org/spec/1.2.2/#3222-anchors-and-aliases
    """

    def ignore_aliases(self, data: Any) -> bool:
        return True


_T = TypeVar("_T")


def sort_dict(d: _T) -> _T:
    """Recursively applicable sorting helper.

    Sorts dictionary-type objects by key, and returns others without changes.
    """
    if isinstance(d, dict):
        return {k: sort_dict(v) for k, v in sorted(d.items())}  # type: ignore
    return d


class YAML(yaml.YAML):
    """Convenience wrapper around ruamel.yaml.YAML configuration object."""

    def __init__(
        self,
        *,
        typ: str = "unsafe",
        default_flow_style: Optional[bool] = False,
    ):
        """Construct the YAML configuration object.

        For argument semantics, see ruamel.yaml.YAML:
        https://sourceforge.net/p/ruamel-yaml/code/ci/default/tree/main.py#l53.
        """
        super().__init__(typ=typ, pure=True)
        self.default_flow_style = default_flow_style
        self.Representer = NoAliasRepresenter
        self.indent(mapping=2, sequence=4, offset=2)

    def dump(
        self,
        data: Any,
        stream: Union[Path, StreamType, None] = None,
        *,
        sort: bool = True,
        **kwargs: Any,
    ):
        """YAMLize the data into a stream.

        Args:
            data: object to be serialized.
            stream: file (as path) or stream object to output the yaml into.
            sort: if dictionaries should be sorted by keys.
            **kwargs: see ruamel.yaml.YAML.dump.

        """
        super().dump(sort_dict(data) if sort else data, stream, **kwargs)


def represent_as_string(
    classes: Iterable[type[Any]],
    representer: type[yaml.BaseRepresenter] = NoAliasRepresenter,
):
    """Configure the yaml parser to serialize classes as strings.

    Args:
        classes: The class objects (not instances) to represent as strings.
        representer: representer class (not instance) to register classes with.
    """

    def _represent_as_string(tag, mapping, flow_style=None):
        return tag.represent_str(str(mapping))

    for cls in classes:
        representer.add_representer(cls, _represent_as_string)


represent_as_string([PosixPath, WindowsPath])
