##===- CMakeLists.txt -----------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_mlir_dialect(Zap zap -dialect=zap)
add_modular_doc(Zap -gen-dialect-doc ZapDialect KGEN/docs/ -dialect=zap)
