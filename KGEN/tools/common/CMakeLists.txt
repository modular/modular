##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(MojoToolsCommon
  Telemetry.cpp

  LINK_LIBS PRIVATE
  MSupportInit
  MSupport
  LLVMOption
)
