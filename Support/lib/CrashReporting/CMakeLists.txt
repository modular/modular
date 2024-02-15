##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(MCrashReporting
  CrashReporting.cpp

  LINK_LIBS PUBLIC
  MSupport
  MVersion
  MSupportSettings

  client  # Crashpad client library
)
