##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

# ASAN triggers errors because of python, so we skip.
if (LLVM_USE_SANITIZER)
  return()
endif()

add_test(NAME mojo-debug-pytests
  COMMAND ${CMAKE_COMMAND} -E env
    LLVM_PROFILE_FILE=${CMAKE_BINARY_DIR}/profiles/mojo-debug-pytests%4m.profraw
    LLDB_TEST_INIT_FILE=${LLDB_TEST_INIT_FILE}
    ${Python3_EXECUTABLE} -m pytest -rs -v ${CMAKE_CURRENT_SOURCE_DIR} --junitxml=junit/test-suite-mojo-debug-pytests-test-results.xml

)
add_custom_target(check-mojo-debug-pytests
  COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -V -R mojo-debug-pytests
  DEPENDS
    "${GENERATED_TESTS_BASE}"
    mojo-sdk
    mojo
)

add_dependencies(check-mojo-debug check-mojo-debug-pytests)
