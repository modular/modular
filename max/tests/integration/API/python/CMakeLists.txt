##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

include(AddPython)
#
# TODO(#8161) Copied over from TensorFlow/unittests/API/python/CMakeLists.txt
# Fixing #8161 would help reduce this duplication.

modular_find_python3_development()

include(AddPyBind11)

# Set up ASAN only if enabled since otherwise errors occur in non-ASAN builds.
if (LLVM_USE_SANITIZER)
  set(PYTHON_API_TESTS_ASAN_ENV "LSAN_OPTIONS=detect_leaks=0")
  if (UNIX)
    list(APPEND PYTHON_API_TESTS_ASAN_ENV "${ASAN_LIB_INJECT_ENV}")
  endif()
endif()


# Add API Scenario tests for the Modular Python API.
add_test(NAME ApiScenarioTest
  COMMAND ${CMAKE_COMMAND} -E env
    ${PYTHON_API_TESTS_ASAN_ENV}
    LLVM_PROFILE_FILE=${CMAKE_BINARY_DIR}/profiles/ApiScenarioTest%4m.profraw
    ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/pytests
)

add_custom_target(check-ApiScenarioTest
  COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -R ApiScenarioTest
  DEPENDS "${GENERATED_TESTS_BASE}"
)

if(TARGET modular-pybind-ext AND ENABLE_API_SCENARIO_TESTING)
  add_dependencies(check-ApiScenarioTest modular-pybind-ext)
endif()
