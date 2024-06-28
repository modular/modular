##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_mlir_dialect(Custom custom -dialect=custom)
add_modular_doc(Custom -gen-dialect-doc CustomDialect KGEN/docs/ -dialect=custom)
