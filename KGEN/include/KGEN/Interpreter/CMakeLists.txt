##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

set(LLVM_TARGET_DEFINITIONS InterpreterInterface.td)
mlir_tablegen(InterpreterOpInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(InterpreterOpInterface.cpp.inc -gen-op-interface-defs)
mlir_tablegen(MemoryableTypeInterface.h.inc -gen-type-interface-decls)
mlir_tablegen(MemoryableTypeInterface.cpp.inc -gen-type-interface-defs)
add_public_tablegen_target(InterpreterInterfaceIncGen)
