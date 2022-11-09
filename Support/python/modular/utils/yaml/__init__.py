# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

__doc__ = """
YAML utility module

This module contains functionality to help us deal with YAML files uniformly.
The implementation relies on ruamel.yaml, which is exposed in various ways, but
users are encouraged to depend on it as little as possible (e.g. by not
importing ruamel.yaml).
"""

from ._yaml import YAML, represent_as_string

# Remove from the namespace so that it's not visible to users.
del _yaml  # noqa: F821
