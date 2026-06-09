# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# This file is imported by 'import.mojo' as part of testing import functionality,
# and does not include any useful testing by itself.


def imported_fn():
    return


def _ignored_wildcard_fn():
    return


# Intentionally the same name as the package.
def imported_module():
    pass
