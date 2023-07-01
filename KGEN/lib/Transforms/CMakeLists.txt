##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_conversion_library(KGENTransforms
  Canonicalizer.cpp
  CleanupCompilerGlobals.cpp
  EliminateDeadSymbols.cpp
  FoldGlobalConstLoads.cpp
  HoistTrivialInvariants.cpp
  Inlining.cpp
  KnownVariant.cpp
  LiftAndFoldApply.cpp
  LoopUnrolling.cpp
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
