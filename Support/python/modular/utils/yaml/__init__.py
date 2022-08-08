##=== __init__.py ---------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

from ._yaml import YAMLNoAliasDumper, represent_as_string, yaml

# Remove from the namespace so that it's not visible to users.
del _yaml
