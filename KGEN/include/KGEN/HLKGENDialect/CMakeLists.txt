##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

# Generating HLKGEN filename prefix into the "kgen" C++ namespace.
add_mlir_dialect(HLKGEN hlkgen)
add_modular_doc(HLKGEN -gen-dialect-doc HLKGENDialect KGEN/docs/ -dialect=hlkgen)

set(LLVM_TARGET_DEFINITIONS HLKGEN.td)
