##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(MSupportInit
  Init.cpp

  LINK_LIBS PUBLIC
  LLCLRuntime
  LLCLRuntimeGlobals
  MCrashReporting
  MSupport
  MSupportEntitlements
  MSupportSettings
  MTelemetry
)
