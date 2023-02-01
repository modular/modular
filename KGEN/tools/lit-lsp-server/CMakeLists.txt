##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_tool(lit-lsp-server
  LSPServer.cpp
  LITServer.cpp
  lit-lsp-server.cpp

  LINK_LIBS PRIVATE
  ParseLit
  MLIRLspServerSupportLib
)
