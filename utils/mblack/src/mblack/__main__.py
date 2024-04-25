# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# ===----------------------------------------------------------------------=== #
#
# File originates from:
#   Repo:   git@github.com:psf/black.git
#   Commit: d4a85643a465f5fae2113d07d22d021d4af4795a
#   Path:   src/black/__main__.py
#
# ===----------------------------------------------------------------------=== #
from mblack import patched_main
import os

if directory := os.getenv("BUILD_WORKSPACE_DIRECTORY"):
    os.chdir(directory)

patched_main()
