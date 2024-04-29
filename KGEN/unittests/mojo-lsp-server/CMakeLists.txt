##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_modular_unittest(
  TOP_LEVEL_CHECK
  check-mojo-lsp-server

  SUITE_NAME
  MojoLSPServerUnitTests

  TEST_NAME
  MojoLSPServerTests

  CodeActionTest.cpp
  CodeBlockTest.cpp
  CompletionTest.cpp
  DiagnosticsTest.cpp
  DocumentSymbolTest.cpp
  SampleTest.cpp
  Support.cpp

  DEPENDS
  mojo-sdk

  LINK_LIBS PRIVATE
  mojo-lsp-test-client
)
