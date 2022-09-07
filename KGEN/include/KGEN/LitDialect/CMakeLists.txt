##===- CMakeLists.txt -----------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_mlir_dialect(Lit lit -dialect=lit)
add_modular_doc(Lit -gen-dialect-doc LitDialect KGEN/docs/ -dialect=lit)
