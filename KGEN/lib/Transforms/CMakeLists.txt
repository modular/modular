##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(KGENTransforms
  ApplyInliner.cpp
  AutomaticInline.cpp
  Canonicalizer.cpp
  DeadArgumentElimination.cpp
  EliminateDeadSymbols.cpp
  FoldGlobalConstLoads.cpp
  HoistTrivialInvariants.cpp
  Inlining.cpp
  LiftAndFoldApply.cpp
  LoopUnrolling.cpp
  LowerClosures.cpp
  LowerLoops.cpp
  LowerArgConventions.cpp
  LowerCallingConventions.cpp
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
  CODialect
  KGENTransformUtils
  HLCFAnalysis
  KGENDialect
  POPDialect
  LITDialect
  MLIRTransformUtils
  MSupport
  LLCLCompilerSupport
)
