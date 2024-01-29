##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(LowerLIT
  CheckLifetimes.cpp
  CheckRecursiveStructs.cpp
  ConstraintSet.cpp
  LowerLIT.cpp
  LowerSemanticCF.cpp
  LowerLITTypes.cpp
  StripParserMetadata.cpp

  DEPENDS
  MLIRKGENPasses

  LINK_LIBS PUBLIC
  KGENDialect
  LITDialect
  DebugInfoTransforms
  MLIRIR
  )
