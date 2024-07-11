##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_tool(crash-test-dummy
  crash-test-dummy.cpp

  LINK_LIBS PRIVATE
  MCrashReporting
  MAsyncRTInit
  MSupportGlobals
  AsyncRTRuntimeGlobals
)
