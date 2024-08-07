# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

__doc__ = """
Miscellaneous utilities

This module can be thought of as a staging ground for utilities that don't fit
into any other existing module yet, and is expected to have more flux than other
utility modules. That is not to say that `misc` is experimental; one motivation
for this utility library is to lift code from experimental to trusted. The
functionality here is expected to be tested, but also to potentially move (at
which point compatibility concerns will be addressed).
"""

from ._misc import (
    create_dir_symlink,
    create_symlink,
    get_ordinal,
    has_gpu,
    modular_dtype_to_np_dtype,
    set_env_var,
)

# Remove from the namespace so that it's not visible to users.
del _misc  # noqa: F821
