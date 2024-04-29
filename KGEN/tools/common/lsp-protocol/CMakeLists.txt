##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(LSPProtocol
  Protocol.cpp

  LINK_LIBS PRIVATE
  MLIRLspServerSupportLib
)
