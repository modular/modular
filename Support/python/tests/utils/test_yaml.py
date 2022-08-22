##=== test_yaml.py --------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##


from enum import Enum
from pathlib import Path, PosixPath, WindowsPath

from ruamel.yaml.compat import StringIO

from modular.utils.typing import Any
from modular.utils.yaml import YAML, represent_as_string

EXPECTED_NO_ALIAS_DUMP = """a_dict:
  a_value: 123
  ref: [a, b, c]
another_dict:
  same_ref: [a, b, c]
  same_value: 123
"""


def _dump_str(data: Any, *, sort: bool = True) -> str:
    stream = StringIO()
    YAML().dump(data, stream, sort=sort)
    return stream.getvalue()


def test_YAML_dumps():
    ref_list = ["a", "b", "c"]
    test_dict = {
        "a_dict": {"a_value": 123, "ref": ref_list},
        "another_dict": {"same_value": 123, "same_ref": ref_list},
    }
    yaml_str = _dump_str(test_dict)
    assert yaml_str == EXPECTED_NO_ALIAS_DUMP


def test_represent_as_string():
    class SomeData(Enum):
        def __str__(self) -> str:
            return self.value

        FOO = "foo"
        BAR = "Bar"

    represent_as_string([SomeData, PosixPath, WindowsPath])

    test_list = [Path("some/posix/path"), SomeData.FOO, SomeData.BAR]
    yaml_str = _dump_str(test_list)
    assert yaml_str == "[some/posix/path, foo, Bar]\n"
