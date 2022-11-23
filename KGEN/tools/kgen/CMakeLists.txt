##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_tool(kgen
  EmitFuncHeader.cpp
  kgen.cpp

  LINK_LIBS PRIVATE
  DebugInfoToLLVM
  DebugInfoTransforms
  KGENCLOptions
  LITDialect
  KGENCompilerRT
  KGENDialect
  KGENElaborator
  KGENTransforms
  ZAPDialect
  LowerLIT
  LowerZAPToPOP
  POPDialect

  MLIRBytecodeWriter
  MLIRIndexDialect
  MLIRParser
)
