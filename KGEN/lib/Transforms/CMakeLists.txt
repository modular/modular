##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_conversion_library(KGENTransforms
  CleanupCompilerGlobals.cpp
  EliminateDeadSymbols.cpp
  Inlining.cpp
  KnownVariant.cpp
  Mem2Reg.cpp
  OutlineClosures.cpp
  VerifyParameters.cpp

  DEPENDS
  MLIRKGENPasses

  LINK_LIBS PUBLIC
  KGENDialect
  POPDialect
)
