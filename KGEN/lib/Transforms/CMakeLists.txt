##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(KGENTransforms
  AutomaticInline.cpp
  Canonicalizer.cpp
  ControlFlowUtils.cpp
  DeadArgumentElimination.cpp
  EliminateDeadSymbols.cpp
  ExternalizePrecompiledFunctions.cpp
  FoldGlobalConstLoads.cpp
  HoistTrivialInvariants.cpp
  Inlining.cpp
  InliningUtils.cpp
  KnownVariant.cpp
  LiftAndFoldApply.cpp
  LoopUnrolling.cpp
  LowerClosures.cpp
  LowerLoops.cpp
  LowerInputConventions.cpp
  MaterializePackages.cpp
  Mem2Reg.cpp
  OutlineClosures.cpp
  RaiseForLoops.cpp
  ResolveCompilerPromises.cpp
  SCCP.cpp
  SimplifyCF.cpp
  SROA.cpp
  StackReuse.cpp
  SyntheticDebugInfo.cpp
  VerifyParameters.cpp

  DEPENDS
  MLIRKGENPasses

  LINK_LIBS PUBLIC
  HLCFAnalysis
  KGENDialect
  POPDialect
  LITDialect
  MLIRTransformUtils
)
