# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Expose `use_foo` (which imports `foo` along two paths) as the package's public
# surface.

from .consumer import use_foo
