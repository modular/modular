##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_tool(mojo-lsp-server
  LSPServer.cpp
  MojoServer.cpp
  mojo-lsp-server.cpp

  LINK_LIBS PRIVATE
  KGENCLOptions
  LLCLRuntimeGlobals
  MArchTarget
  MSupportGlobals
  MojoParser
  MLIRLspServerSupportLib
  MLIRTransformUtils
)
