##===- CMakeLists.txt -----------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_mlir_dialect(ZAP zap -dialect=zap)
add_modular_doc(ZAP -gen-dialect-doc ZAPDialect KGEN/docs/ -dialect=zap)
