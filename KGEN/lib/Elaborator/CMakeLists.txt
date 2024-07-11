##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(KGENElaborator
  Elaborator.cpp
  IREvaluator.cpp

  DEPENDS
  KGENVersion
  MLIRKGENPasses

  LINK_LIBS PUBLIC
  DebugInfoDialect
  InterpreterInterface
  KGENDialect
  KGENSupport
  LITDialect
  MArchTarget

  CustomDialect
  MLIRIndexDialect
  MLIRSupport
  MSupport
  AsyncRTCompilerSupport
)
