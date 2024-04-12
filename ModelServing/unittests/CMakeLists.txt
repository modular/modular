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

# TODO(yihualou): Temporarily disable tests while we renovate.
# add_modular_unittest(
#   SUITE_NAME
#   MuttUnitTests

#   TEST_NAME
#   MuttTests

#   server_test.cpp
#   test_mutt.cpp
#   test_server_async.cpp
#   test_server.cpp

#   LINK_LIBS PRIVATE
#   ModularMuttLib
#   tf

#   TAGS
#   asan-red
# )

# target_compile_definitions(MuttTests
#   PRIVATE
#   MODULAR_MODEL_DIR="${CMAKE_BINARY_DIR}/GeneratedTests/BINARIES/models"
# )

# add_dependencies(MuttTests ExtractGenModels)
