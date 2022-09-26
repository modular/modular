##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_tool(kgen-opt
  kgen-opt.cpp
  LINK_LIBS PRIVATE
  HLKGENDialect
  IndexDialect
  IndexToLLVM
  KGENDialect
  ZAPDialect
  POPDialect
  KGENToLLVM
  KGENElaborator
  LowerHLKGEN
  LowerZAPToPOP

  MLIRIR
  MLIRLLVMDialect
  MLIROptLib
  MLIRParser
  MLIRSCFDialect
  MLIRSupport
  MLIRTransforms
  )
