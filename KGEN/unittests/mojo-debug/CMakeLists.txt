##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_modular_unittest(
  TOP_LEVEL_CHECK
  check-mojo-debug

  SUITE_NAME
  MojoDebugUnitTests

  TEST_NAME
  MojoDebugTests

  BoolTest.cpp
  BreakpointTest.cpp
  SourceNamesTest.cpp
  Support.cpp

  DEPENDS
  mojo-sdk

  LINK_LIBS PRIVATE
  liblldb
  KGENSupport
  MSupport
)
