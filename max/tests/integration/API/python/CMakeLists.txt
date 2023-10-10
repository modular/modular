##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

# Set up ASAN only if enabled since otherwise errors occur in non-ASAN builds.
if (LLVM_USE_SANITIZER)
  # Set environment variables to inject the ASAN runtime library since the
  # Python interpreter is not linked with ASAN.
  # Also, set LSAN_OPTIONS=detect_leaks=0 to avoid leaks exposed by Python.
  set(PYTHON_API_TESTS_ASAN_ENV "LSAN_OPTIONS=detect_leaks=0")

  # TODO(#10762): Fix this with a more robust solution that does not rely on
  # LD_LIBRARY_PATH.
  set(LIBRARY_PATH_VAR_NAME
    $<$<STREQUAL:${CMAKE_SYSTEM_NAME},Darwin>:DYLD_LIBRARY_PATH>$<$<STREQUAL:${CMAKE_SYSTEM_NAME},Linux>:LD_LIBRARY_PATH>)
  list(APPEND PYTHON_API_TESTS_ASAN_ENV
    "${LIBRARY_PATH_VAR_NAME}=$<TARGET_FILE_DIR:mof>")

  if (UNIX)
    list(APPEND PYTHON_API_TESTS_ASAN_ENV "${ASAN_LIB_INJECT_ENV}")
  endif()
endif()

# Add Python tests for the Modular Python API.
# DYLD_INSERT_LIBRARIES is passed as a CMake cache variable so that pytest can
# run when configured as ASAN.
# Without this, on MacOS with ASAN, pytest fails with an error.
add_test(NAME ModularEnginePythonApiTests
  COMMAND ${CMAKE_COMMAND} -E env
    ${PYTHON_API_TESTS_ASAN_ENV}
    # The LLVM profile name has to be unique for ModularEnginePythonApiTests because
    # it links against shared libs that have been patched.
    # Otherwise, the call to llvm-profdata merge will fail.
    LLVM_PROFILE_FILE=${CMAKE_BINARY_DIR}/profiles/ModularEnginePythonApiTests%4m.profraw
    ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/pytests
)

add_custom_target(check-ModularEnginePythonApiTests
  COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -R ModularEnginePythonApiTests

  DEPENDS
  mof
)

if(TARGET modular-pybind-ext)
  add_dependencies(check-ModularEnginePythonApiTests modular-pybind-ext)
  add_dependencies(check-api check-ModularEnginePythonApiTests)
endif()
