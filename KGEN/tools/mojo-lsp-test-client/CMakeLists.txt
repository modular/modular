##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(mojo-lsp-test-client
  Document.cpp
  LSPBatchClient.cpp
  Protocol.cpp

  LINK_LIBS PRIVATE
  MLIRLspServerSupportLib
  MSupport
  KGENSupport
)
