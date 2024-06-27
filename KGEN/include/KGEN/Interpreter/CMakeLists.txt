##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_mlir_dialect(Interpreter interp -dialect=interp)
add_modular_doc(Interpreter -gen-dialect-doc InterpreterDialect KGEN/docs/ -dialect=interp)

set(LLVM_TARGET_DEFINITIONS InterpreterAttrs.td)
mlir_tablegen(InterpreterAttrs.h.inc -gen-attrdef-decls -dialect=interp)
mlir_tablegen(InterpreterAttrs.cpp.inc -gen-attrdef-defs -dialect=interp)
add_public_tablegen_target(MLIRInterpreterAttrsIncGen)
add_dependencies(MLIRInterpreterIncGen MLIRInterpreterAttrsIncGen)

set(LLVM_TARGET_DEFINITIONS InterpreterDialectBytecode.td)
mlir_tablegen(InterpreterDialectBytecode.cpp.inc -gen-bytecode -bytecode-dialect=interp)
add_public_tablegen_target(InterpreterDialectBytecodeIncGen)
add_dependencies(MLIRInterpreterIncGen InterpreterDialectBytecodeIncGen)

set(LLVM_TARGET_DEFINITIONS InterpreterInterface.td)
mlir_tablegen(InterpreterOpInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(InterpreterOpInterface.cpp.inc -gen-op-interface-defs)
mlir_tablegen(MemoryableTypeInterface.h.inc -gen-type-interface-decls)
mlir_tablegen(MemoryableTypeInterface.cpp.inc -gen-type-interface-defs)
add_public_tablegen_target(InterpreterInterfaceIncGen)
add_dependencies(InterpreterInterfaceIncGen MLIRInterpreterIncGen)
