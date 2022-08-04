#===- CMakeLists.txt ------------------------------------------*- cmake -*-===##
#
# This file is Modular Inc proprietary.
#
#===-----------------------------------------------------------------------===##

# Generating POP filename prefix into the "kgen" C++ namespace.
add_mlir_dialect(POP pop -dialect=pop)
add_modular_doc(POP -gen-dialect-doc POPDialect KGEN/docs/ -dialect=pop)
