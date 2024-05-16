##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(MojoBuild
  BSPClient.cpp
  BSPServer.cpp
  MojoBuildServer.cpp
  Protocol.cpp
  Server.cpp

  LINK_LIBS PUBLIC
  KGENSupport
  MSupport
  MVersion

  MLIRLspServerSupportLib
  MLIRSupport
)
