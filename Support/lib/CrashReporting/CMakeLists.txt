##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(MCrashReporting
  CrashReporting.cpp

  LINK_LIBS PUBLIC
  MSupport

  client  # Crashpad client library
)
