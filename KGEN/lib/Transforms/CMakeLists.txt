##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_conversion_library(KGENTransforms
  CleanupCompilerGlobals.cpp
  EliminateDeadSymbols.cpp
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
