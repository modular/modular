##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_tool(kgen-opt
  kgen-opt.cpp

  LINK_LIBS PRIVATE
  DebugInfoToLLVM
  LITDialect
  KGENDialect
  KGENEmitObject
  KGENCompilerRT
  POPDialect
  KGENToLLVM
  KGENElaborator
  KGENTransforms
  LowerLIT

  MLIRIR
  MLIRIndexDialect
  MLIRIndexToLLVM
  MLIRLLVMDialect
  MLIROptLib
  MLIRParser
  MLIRReconcileUnrealizedCasts
  MLIRSupport
  MLIRTransforms
  )
