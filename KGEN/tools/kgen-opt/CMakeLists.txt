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
  ZapDialect
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
