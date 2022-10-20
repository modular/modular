# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os
from contextlib import contextmanager

from modular.utils.typing import Iterator, Optional


@contextmanager
def set_env_var(env_var: str, tmp_val: Optional[str]) -> Iterator[None]:
    """Temporarily set an environment variable to a different value.

    This is modeled as a context manager that yields nothing.

    Args:
        env_var: name of the environment variable as a string.
        tmp_val: temporary value for the variable in the context. If None, the
            variable will be temporarily removed.
    """

    # Store the old value, and remove the environment variable if it exists.
    old_val = os.environ.pop(env_var, None)

    # If a value is given, we set the variable to that value
    if tmp_val is not None:
        os.environ[env_var] = tmp_val

    try:
        yield
    finally:
        if old_val is None:
            # Delete if it didn't exist before and exists now.
            os.environ.pop(env_var, None)
        else:
            # We reset to the original value.
            os.environ[env_var] = old_val
