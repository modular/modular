##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_tool(kgen-opt
  kgen-opt.cpp
  LINK_LIBS PRIVATE
  LITDialect
  IndexDialect
  IndexToLLVM
  KGENDialect
  KGENEmitObject
  ZAPDialect
  POPDialect
  KGENToLLVM
  KGENElaborator
  KGENTransforms
  LowerLIT
  LowerZAPToPOP

  MLIRIR
  MLIRLLVMDialect
  MLIROptLib
  MLIRParser
  MLIRReconcileUnrealizedCasts
  MLIRSCFDialect
  MLIRSupport
  MLIRTransforms
  )
