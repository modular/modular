# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# ===----------------------------------------------------------------------=== #
#
# This exercises the parser through the language-server parse path
# (`parseFileForLSP`) via `kgen-translate -import-mojo -lsp`, rather than the
# regular compiler `importMojoFile` path.
#
# Besides checking that the user's own decl is emitted, this asserts on output
# that is unique to LSP mode: `lit.unresolved_import` ops. These are the lazy
# named imports that `resolveSignaturesForLSP` deliberately leaves unparsed, and
# that the LSP path preserves because it skips DCE (`eraseUnreachableDecls`).
# The regular compiler path resolves and then strips all imports, so it never
# emits a `lit.unresolved_import`. A second RUN confirms that absence, so the
# test fails if `-lsp` silently falls back to the compiler path.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -lsp %s | FileCheck %s
# RUN: %parse-mojo-isolated %s | FileCheck %s --check-prefix=REGULAR

# CHECK: lit.struct.decl @BoxedInt
# REGULAR: lit.struct.decl @BoxedInt
@fieldwise_init
struct BoxedInt(ImplicitlyCopyable):
    var value: Int


def main():
    pass

# LSP mode keeps the lazy named imports as unresolved-import ops; the compiler
# path strips them via DCE.
# CHECK: lit.unresolved_import @".builtin" as @builtin
# REGULAR-NOT: lit.unresolved_import
