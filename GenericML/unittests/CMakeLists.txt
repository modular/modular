##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_unittest(
  TOP_LEVEL_CHECK
  check-genericml

  SUITE_NAME
  GenericMLUnitTests

  TEST_NAME
  GenericMLTests

  API/c/APITest.cpp
  BufferRefTest.cpp
  ContextTest.cpp
  TensorTest.cpp

  LINK_LIBS PRIVATE
  GenericMLSupport
  LLCLRuntime
  MSupportML
  modular-framework
)

if(MODULAR_BUILD_PYTHON_API_BINDINGS)
  add_subdirectory(API/python)
endif()
