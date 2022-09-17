##===- CMakeLists.txt -----------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_tool(kgen
  EmitFuncHeader.cpp
  EmitFuncObject.cpp
  kgen.cpp

  LINK_LIBS PRIVATE
  KGENCLOptions
  HLKGENDialect
  IndexDialect
  KGENCompilerRT
  KGENElaborator
  KGENDialect
  ZAPDialect
  LowerHLKGEN
  LowerZAPToPOP
  MetaDialect
  POPDialect

  MLIRParser
)
