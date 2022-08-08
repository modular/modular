##=== _yaml.py ------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

from ruamel import yaml

from modular.utils.typing import Any, Iterable, Type


class YAMLNoAliasDumper(yaml.Dumper):
    """A custom YAML dumper that ignores aliases.

    Aliases are defined by the YAML spec
    https://yaml.org/spec/1.2.2/#3222-anchors-and-aliases
    """

    def ignore_aliases(self, data: Any) -> bool:
        return True


def represent_as_string(classes: Iterable[Type[Any]]):
    """Configure the yaml parser to serialize classes as strings.

    Args:
        classes: The class objects (not instances) to represent as strings.
    """

    def _represent_as_string(tag, mapping, flow_style=None):
        return tag.represent_str(str(mapping))

    for cls in classes:
        yaml.add_representer(cls, _represent_as_string)


# TODO: uncomment below when modular-source.py uses represent_as_string
# represent_as_string([PosixPath, WindowsPath])
