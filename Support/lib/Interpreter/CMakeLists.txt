##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(InterpreterInterface
  InterpreterInterface.cpp

  DEPENDS
  InterpreterInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MSupport
  MCompilerSupport
  MDialect
  MLIRCallInterfaces
)
