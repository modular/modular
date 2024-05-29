##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(KGENJIT
  JITSupport.cpp
  MaterializationLayer.cpp
  ObjectCompilerLayer.cpp
  StaticArchiveLayer.cpp

  DEPENDS
  KGENVersion

  LINK_COMPONENTS
  AllTargetsInfos
  OrcJIT
  JITLink
  MC

  LINK_LIBS PUBLIC
  KGENToolCommon

  MCache
  MCompilerSupport
  MTelemetry
  MSupport
  LLCLCompilerSupport

  MLIRTargetLLVMIRExport
)
