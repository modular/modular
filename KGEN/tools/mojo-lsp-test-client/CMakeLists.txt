##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(mojo-lsp-test-client
  Document.cpp
  LSPBatchClient.cpp

  LINK_LIBS PRIVATE
  LSPProtocol
  MLIRLspServerSupportLib
  MSupport
  KGENSupport
)
