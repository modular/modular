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
  KGENToLLVM
  KGENTransforms
  LowerLIT

  AsyncRTCompilerSupport
  AsyncRTRuntimeGlobals
  MSupportGlobals
  MAsyncRTInit
  MDialect

  MLIRIR
  MLIRIndexToLLVM
  MLIROptLib
  MLIRParser
  MLIRSupport
  MLIRTransforms
  )
