##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

set(LLVM_TARGET_DEFINITIONS InterpreterInterface.td)
mlir_tablegen(InterpreterInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(InterpreterInterface.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(InterpreterInterfaceIncGen)
