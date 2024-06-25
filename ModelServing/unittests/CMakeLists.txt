##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_custom_target(ExtractGenModels
  DEPENDS ${CMAKE_BINARY_DIR}/GeneratedTests/BINARIES/models/BasicAdd_float32/config.pbtxt
)
add_custom_command(PRE_BUILD
  # COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/GeneratedTests/BINARIES/
  COMMAND ${CMAKE_COMMAND} -E tar xzfv ${CMAKE_SOURCE_DIR}/GeneratedTests/BINARIES.tar.gz
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/GeneratedTests/
  DEPENDS ${CMAKE_SOURCE_DIR}/GeneratedTests/BINARIES.tar.gz

  # Can be any generated file; allows us to skip generation if nothing changed.
  OUTPUT ${CMAKE_BINARY_DIR}/GeneratedTests/BINARIES/models/BasicAdd_float32/config.pbtxt
  COMMENT "Extracting generated models"
  VERBATIM
)

include_directories(PRIVATE ${CMAKE_SOURCE_DIR}/ModelServing/include/)

add_modular_unittest(
  SUITE_NAME
  ServeRTUnitTests

  TEST_NAME
  ServeRTTests

  test_server_async.cpp

  LINK_LIBS PRIVATE
  ServeRTLib
)

target_compile_definitions(ServeRTTests
  PRIVATE
  MODULAR_MODEL_DIR="${CMAKE_BINARY_DIR}/GeneratedTests/BINARIES/models"
)

add_dependencies(ServeRTTests ExtractGenModels)
