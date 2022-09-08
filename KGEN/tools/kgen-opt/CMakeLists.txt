##===- CMakeLists.txt -----------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_tool(kgen-opt
  kgen-opt.cpp
  LINK_LIBS PRIVATE
  HLKGENDialect
  IndexDialect
  KGENDialect
  LitDialect
  MetaDialect
  POPDialect
  KGENToLLVM
  KGENElaborator
  LowerHLKGEN

  MLIRIR
  MLIRLLVMDialect
  MLIROptLib
  MLIRParser
  MLIRSCFDialect
  MLIRSupport
  MLIRTransforms
  )
