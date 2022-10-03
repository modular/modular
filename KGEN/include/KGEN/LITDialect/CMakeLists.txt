##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

# Generating LIT filename prefix into the "kgen" C++ namespace.
add_mlir_dialect(LIT lit)
add_modular_doc(LIT -gen-dialect-doc LITDialect KGEN/docs/ -dialect=lit)

set(LLVM_TARGET_DEFINITIONS LIT.td)
