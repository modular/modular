##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(MSupportSettings
  Settings.cpp

  LINK_LIBS PUBLIC
  MSupport
  MSupportEntitlements
)
