##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(InterpreterInterface
  InterpreterAttrs.cpp
  InterpreterDialect.cpp
  InterpreterInterface.cpp
  InterpreterState.cpp

  DEPENDS
  InterpreterInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MSupport
  MCompilerSupport
  MDialect
  MLIRCallInterfaces
)
