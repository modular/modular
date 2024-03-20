##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_tool(mojo-lsp-server
  LSPServer.cpp
  LSPTelemetryContext.cpp
  MojoServer.cpp
  mojo-lsp-server.cpp
  Protocol.cpp
  SemanticTokens.cpp

  LINK_LIBS PRIVATE
  KGENPipeline
  KGENToolCommon
  LLCLRuntimeGlobals
  MArchTarget
  MSupportGlobals
  MojoParser
  MojoTooling
  MLIRLspServerSupportLib
  MLIRTransformUtils
)
