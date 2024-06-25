##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_tool(system-info
  system-info.cpp

  LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsInfos

  LINK_LIBS PRIVATE
  MSupport
  MProfiling
  MSupportGlobals
  MArchTarget
)
