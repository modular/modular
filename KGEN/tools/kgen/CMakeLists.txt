##===- CMakeLists.txt -----------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_tool(kgen
  EmitKernelHeader.cpp
  EmitKernelObject.cpp
  kgen.cpp

  LINK_LIBS PRIVATE
  KGENCLOptions
  HLKGENDialect
  KGENKernelElaborator
  KGENDialect
  LowerHLKGEN
  MetaDialect
  POPDialect

  MLIRParser
)
