##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_tool(crash-report-path-info
  crash-report-path-info.cpp

  LINK_LIBS PRIVATE
  MCrashReporting
  MSupportInit
  MSupportGlobals
  LLCLRuntimeGlobals
)
