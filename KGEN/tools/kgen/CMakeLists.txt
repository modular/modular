##===- CMakeLists.txt -----------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_tool(kgen
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
