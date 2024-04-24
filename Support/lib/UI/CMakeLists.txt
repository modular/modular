##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##


add_modular_library(MSupportUI
  STATIC OBJECT # Always generate an obj.${name} target.

  ProgressBar.cpp
  DataProgressBar.cpp
  IndeterminateProgressBar.cpp

  LINK_LIBS PUBLIC
  MSupport
  )
