##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

# Generating POP filename prefix into the "kgen" C++ namespace.
add_mlir_dialect(POP pop -dialect=pop)
add_modular_doc(POP -gen-dialect-doc POPDialect KGEN/docs/ -dialect=pop)

set(LLVM_TARGET_DEFINITIONS POP.td)
mlir_tablegen(POPAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=pop)
mlir_tablegen(POPAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=pop)
add_public_tablegen_target(MLIRPOPAttrsIncGen)
add_dependencies(MLIRPOPIncGen MLIRPOPAttrsIncGen)

set(LLVM_TARGET_DEFINITIONS POPAttrs.td)
mlir_tablegen(POPEnums.h.inc -gen-enum-decls)
mlir_tablegen(POPEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRPOPEnumsIncGen)
add_dependencies(MLIRPOPIncGen MLIRPOPEnumsIncGen)
