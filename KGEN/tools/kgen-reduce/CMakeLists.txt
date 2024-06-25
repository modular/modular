##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_tool(kgen-reduce
  Helpers.cpp
  kgen-reduce.cpp

  LINK_LIBS PRIVATE
  KGENToolCommon
  DebugInfoToLLVM
  KGENCompiler
  KGENElaborator
  KGENToLLVM
  KGENTransforms
  LowerLIT

  LLCLCompilerSupport
  LLCLRuntimeGlobals
  MSupportGlobals
  MLLCLInit
  MDialect

  MLIRIR
  MLIRIndexToLLVM
  MLIROptLib
  MLIRParser
  MLIRSupport
  MLIRTransforms
)
