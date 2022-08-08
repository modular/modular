##=== test_yaml.py --------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

from enum import Enum
from pathlib import Path, PosixPath, WindowsPath

from modular.utils.yaml import YAMLNoAliasDumper, represent_as_string, yaml

EXPECTED_NO_ALIAS_DUMP = """a_dict:
  a_value: 123
  ref: [a, b, c]
another_dict:
  same_ref: [a, b, c]
  same_value: 123
"""


def test_YAMLNoAliasDumper():
    ref_list = ["a", "b", "c"]
    test_dict = {
        "a_dict": {"a_value": 123, "ref": ref_list},
        "another_dict": {"same_value": 123, "same_ref": ref_list},
    }
    yaml_str = yaml.dump(test_dict, Dumper=YAMLNoAliasDumper)
    assert yaml_str == EXPECTED_NO_ALIAS_DUMP


def test_represent_as_string():
    class SomeData(Enum):
        def __str__(self) -> str:
            return self.value

        FOO = "foo"
        BAR = "Bar"

    represent_as_string([SomeData, PosixPath, WindowsPath])

    test_list = [Path("some/posix/path"), SomeData.FOO, SomeData.BAR]
    yaml_str = yaml.dump(test_list)
    assert yaml_str == "[some/posix/path, foo, Bar]\n"
