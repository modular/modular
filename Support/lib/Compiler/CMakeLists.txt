##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_mlir_library(MCompilerSupport
  BytecodeReaderWriter.cpp
  Diags.cpp
  ErrorTree.cpp
  MLIRDenseAttr.cpp
  MLIRDType.cpp
  OperationUtils.cpp
  ParserUtils.cpp
  ThreadDiagnosticHandler.cpp
  TimeProfilerTimingManager.cpp
  VerifyUtils.cpp

  LINK_LIBS PUBLIC
  MDialect
  MSupport
  MSupportML
  MLIRIR
)
