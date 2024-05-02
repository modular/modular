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

  LINK_LIBS PRIVATE
  KGENPipeline
  KGENToolCommon
  LLCLRuntimeGlobals
  LSPProtocol
  MArchTarget
  MSupportGlobals
  MSupportInit
  MojoParser
  MojoTooling
  MLIRLspServerSupportLib
  MLIRTransformUtils
)
