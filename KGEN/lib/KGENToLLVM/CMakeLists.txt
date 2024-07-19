##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(KGENToLLVM
  DebugInfoTypeConverter.cpp
  LLVMLoweringUtils.cpp
  LowerControlFlow.cpp
  LowerKGENClosuresToLLVM.cpp
  LowerKGENToLLVM.cpp
  LowerPOPToLLVM.cpp
  LowerSuspensionPoints.cpp

  DEPENDS
  MLIRKGENPasses

  LINK_LIBS PUBLIC
  CODialect
  DebugInfoToLLVM
  DebugInfoTransforms
  HLCFAnalysis
  KGENDialect
  KGENSupport
  MCompilerSupport
  POPDialect

  MLIRIR
  MLIRIndexToLLVM
  MLIRNVVMToLLVM
  MLIRLLVMCommonConversion
)
