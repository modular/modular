#===- CMakeLists.txt ------------------------------------------*- cmake -*-===##
#
# This file is Modular Inc proprietary.
#
#===-----------------------------------------------------------------------===##

# Generating KGEN filename prefix into the "kgen" C++ namespace.
add_mlir_dialect(KGEN kgen)
add_modular_doc(KGEN -gen-dialect-doc KGENDialect KGEN/docs/)

