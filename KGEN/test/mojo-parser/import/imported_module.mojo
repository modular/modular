# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# This file is imported by 'import.mojo' as part of testing import functionality,
# and does not include any useful testing by itself.


fn imported_fn():
    return


fn _ignored_wildcard_fn():
    return


# Intentionally the same name as the package.
fn imported_module():
    pass
