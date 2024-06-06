##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##


add_modular_library(MThreading
  STATIC OBJECT # Always generate an obj.${name} target.

  HWInfo.cpp
  SpinWaiter.cpp
  ThreadAffinity.cpp

  LINK_LIBS PUBLIC
  MSupport
  )
