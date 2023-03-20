##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_tool(kgen-execute
  kgen-execute.cpp

  LINK_COMPONENTS
  native
  nativecodegen

  LINK_LIBS PRIVATE
  DebugInfoTransforms
  KGENCLOptions
  LLCLRuntimeGlobals
  MLIRParser
  KGENEmitObject
  KGENExecutionEngine
  KGENCompilerRT)
