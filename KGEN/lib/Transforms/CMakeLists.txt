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
  Mem2Reg.cpp
  OutlineClosures.cpp
  SimplifyCF.cpp
  SROA.cpp
  SyntheticDebugInfo.cpp
  VerifyParameters.cpp

  DEPENDS
  MLIRKGENPasses

  LINK_LIBS PUBLIC
  KGENDialect
  POPDialect
)
