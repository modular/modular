# ===- __init__.py --------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from ._yaml import YAML, represent_as_string

# Remove from the namespace so that it's not visible to users.
del _yaml  # noqa: F821
