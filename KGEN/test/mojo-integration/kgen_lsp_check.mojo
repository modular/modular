# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# ===----------------------------------------------------------------------=== #
#
# Exercises the `kgen -lsp` command, which reproduces how the language server
# processes an open document (`MojoDocument::checkModuleSemantics`): an
# error-tolerant parse (`parseFileForLSP`) followed by the module-level check
# pipeline (`runCheckLITPipeline`) on the per-decl clone. The resulting checked
# module IR is printed to stdout (diagnostics, if any, go to stderr).
#
# This asserts that the command emits the checked IR and that the user's own
# declarations from this file survive into it, mirroring how `kgen-translate
# -lsp` is tested under mojo-parser. Patterns are matched loosely so they hold
# whether the module prints in generic or pretty form.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen -lsp %s | FileCheck %s


# CHECK: lit.struct.decl{{.*}}BoxedInt
@fieldwise_init
struct BoxedInt(ImplicitlyCopyable):
    var value: Int


# CHECK: lit.fn{{.*}}get_boxed
def get_boxed() -> BoxedInt:
    return BoxedInt(42)
