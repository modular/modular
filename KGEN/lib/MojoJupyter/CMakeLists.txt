##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(MojoJupyter SHARED
  Kernel.cpp

  LINK_LIBS PRIVATE
  MSupport
  MojoLLDB

  liblldb
)
