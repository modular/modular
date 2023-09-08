##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_mlir_library(MCompilerSupport
  ErrorTree.cpp
  MLIRDType.cpp
  OperationUtils.cpp
  ParserUtils.cpp
  TimeProfilerTimingManager.cpp
  VerifyUtils.cpp

  LINK_LIBS PUBLIC
  MDialect
  MSupport
  MSupportML
  MLIRIR
)
