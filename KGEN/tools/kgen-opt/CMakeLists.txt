##===- CMakeLists.txt -----------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_tool(kgen-opt
  kgen-opt.cpp
  LINK_LIBS PRIVATE
  HLKGENDialect
  KGENDialect
  LitDialect
  MetaDialect
  POPDialect
  KGENToLLVM
  KGENElaborator
  LowerHLKGEN

  MLIRArithmeticDialect
  MLIRIR
  MLIRLLVMDialect
  MLIROptLib
  MLIRParser
  MLIRSCFDialect
  MLIRSupport
  MLIRTransforms
  )
