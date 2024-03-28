##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(MojoBuild
  Server.cpp

  LINK_LIBS PUBLIC
  MLIRLspServerSupportLib
  MLIRSupport
)
