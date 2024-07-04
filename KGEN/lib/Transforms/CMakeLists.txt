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
  FunctionStats.cpp
  HoistTrivialInvariants.cpp
  InlineParametric.cpp
  IPDF.cpp
  LiftAndFoldApply.cpp
  LoopUnrolling.cpp
  LowerAsyncFunctions.cpp
  LowerClosures.cpp
  LowerLoops.cpp
  LowerArgConventions.cpp
  LowerCallingConventions.cpp
  LowerCustomOps.cpp
  MaterializePackages.cpp
  Mem2Reg.cpp
  OutlineClosures.cpp
  RaiseForLoops.cpp
  RemoveUnusedParams.cpp
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
  CustomDialect
  LITDialect
  MLIRTransformUtils
  MSupport
  LLCLCompilerSupport
)
