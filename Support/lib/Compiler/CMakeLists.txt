##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_mlir_library(MCompilerSupport
  BytecodeReaderWriter.cpp
  DiagnosticHandler.cpp
  Diags.cpp
  DomainAwareReplacer.cpp
  ErrorTree.cpp
  MLIRDenseAttr.cpp
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
  MLIRBytecodeReader
  MLIRBytecodeWriter
)
