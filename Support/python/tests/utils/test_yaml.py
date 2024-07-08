# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


from enum import Enum
from pathlib import Path, PosixPath, WindowsPath
from typing import Any

from ruamel.yaml.compat import StringIO

from modular.utils.yaml import YAML, represent_as_string


def _dump_str(data: Any, *, sort: bool = True) -> str:
    stream = StringIO()
    YAML().dump(data, stream, sort=sort)
    return stream.getvalue()


EXPECTED_NO_ALIAS_DUMP = """a_dict:
  a_value: 123
  ref:
    - a
    - b
another_dict:
  same_ref:
    - a
    - b
  same_value: 123
"""


def test_YAML_dumps():
    ref_list = ["a", "b"]
    test_dict = {
        "a_dict": {"a_value": 123, "ref": ref_list},
        "another_dict": {"same_value": 123, "same_ref": ref_list},
    }
    assert _dump_str(test_dict) == EXPECTED_NO_ALIAS_DUMP


EXPECTED_PATHS_DUMP = """  - some/posix/path
  - foo
  - Bar
"""


def test_represent_as_string():
    class SomeData(Enum):
        def __str__(self) -> str:
            return self.value

        FOO = "foo"
        BAR = "Bar"

    represent_as_string([SomeData, PosixPath, WindowsPath])

    test_list = [Path("some/posix/path"), SomeData.FOO, SomeData.BAR]
    assert _dump_str(test_list) == EXPECTED_PATHS_DUMP
