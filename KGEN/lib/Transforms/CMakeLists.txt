##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_conversion_library(KGENTransforms
  Canonicalizer.cpp
  CleanupCompilerGlobals.cpp
  ControlFlowUtils.cpp
  EliminateDeadSymbols.cpp
  ExternalizePrecompiledFunctions.cpp
  FoldGlobalConstLoads.cpp
  HoistTrivialInvariants.cpp
  Inlining.cpp
  InliningUtils.cpp
  KnownVariant.cpp
  LiftAndFoldApply.cpp
  LoopUnrolling.cpp
  LowerLoops.cpp
  Mem2Reg.cpp
  OutlineClosures.cpp
  RaiseForLoops.cpp
  SimplifyCF.cpp
  SROA.cpp
  StackReuse.cpp
  SyntheticDebugInfo.cpp
  VerifyParameters.cpp

  DEPENDS
  MLIRKGENPasses

  LINK_LIBS PUBLIC
  KGENDialect
  POPDialect
)
