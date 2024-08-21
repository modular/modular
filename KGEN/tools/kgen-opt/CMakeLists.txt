##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_tool(kgen-opt
  kgen-opt.cpp

  LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsInfos

  LINK_LIBS PRIVATE
  KGENToolCommon
  DebugInfoToLLVM
  KGENCompiler
  KGENElaborator
  KGENSupportForceLinkMLIRC
  KGENToLLVM
  KGENTransforms
  LowerLIT

  AsyncRTCompilerSupport
  AsyncRTRuntimeGlobals
  MSupportGlobals
  Init
  MDialect

  MLIRIR
  MLIRIndexToLLVM
  MLIROptLib
  MLIRParser
  MLIRSupport
  MLIRTransforms
  )

set_property(TARGET kgen-opt PROPERTY ENABLE_EXPORTS ON)
