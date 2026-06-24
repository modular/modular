# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -I=%S/inputs %s

# Regression test: a plain dotted `import a.b.c` is built into its gated
# ImportOp chain at parse time, so every segment it binds (`a`, `a.b`, `a.b.c`)
# is resolvable before any reference. `timing_user` is resolved on-demand (as a
# dependency, not a top-level script), and uses its dotted imports from default
# arguments — which are resolved before function bodies. This previously failed
# with "use of unknown declaration" because the import's tree was only built
# lazily, after signatures.
from timing_user import body_use, default_use


def main():
    body_use()
    default_use()
