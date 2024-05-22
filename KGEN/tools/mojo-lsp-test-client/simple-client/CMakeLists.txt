##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_tool(mojo-lsp-simple-client
  simple-client.cpp

  LINK_LIBS PRIVATE
  mojo-lsp-test-client
  MSupport
  MSupportGlobals
)
