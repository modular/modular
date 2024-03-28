##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(MojoBuild
  BSPServer.cpp
  Protocol.cpp
  Server.cpp

  LINK_LIBS PUBLIC
  MLIRLspServerSupportLib
  MLIRSupport
)
