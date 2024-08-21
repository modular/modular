##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(Init
  Init.cpp

  LINK_LIBS PUBLIC
  AsyncRTRuntime
  MCrashReporting
  MSupport
  MSupportEntitlements
  MSupportSettings
  MTelemetry
)
