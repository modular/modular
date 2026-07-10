#!/usr/bin/env python3
# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Bazel entry point for lsp-parse-smoke-kernels.

Kept separate from lsp_smoke_all.py (see lsp_smoke_lib) so that script has
exactly one owning rule; this test, lsp-parse-smoke-stdlib, and lsp-smoke-one
each get their own thin `main` wrapper instead of listing the same file as
`main`. See bazel/internal/find_duplicate_srcs.py.
"""

import sys

from lsp_smoke_all import main

if __name__ == "__main__":
    sys.exit(main())
