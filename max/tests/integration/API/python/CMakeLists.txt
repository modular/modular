##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

#
# TODO(#8161) Copied over from TensorFlow/unittests/API/python/CMakeLists.txt
# Fixing #8161 would help reduce this duplication.

include(AddPyEnv)

pyenv_setup_findpython3_variables("${MODULAR_PYTHON_API_PYTHON_VERSION}" "${MODULAR_PYTHON_API_PYTHON_ROOT}")
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

include(AddPyBind11)

# Install dependencies needed for pytest unit tests.
execute_process(
  COMMAND ${Python3_EXECUTABLE} -m pip install -r ${CMAKE_CURRENT_SOURCE_DIR}/pytests/requirements.txt
  OUTPUT_QUIET
  RESULT_VARIABLE PIP_INSTALL_DEPS_RESULT
)
if (PIP_INSTALL_DEPS_RESULT)
  message(FATAL_ERROR "Installation of Python dependencies failed in Python unit test configuration.")
endif()

if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64)$")
  set(HOST_ARCH x86_64)
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
  set(HOST_ARCH aarch64)
else()
  message(FATAL_ERROR "Unsupported target architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

# Set up ASAN only if enabled since otherwise errors occur in non-ASAN builds.
if (LLVM_USE_SANITIZER)
  set(PYTHON_API_TESTS_ASAN_ENV "LSAN_OPTIONS=detect_leaks=0")
  if (UNIX)
    if (APPLE)
      set(INJECT_LIB_ENV_VAR DYLD_INSERT_LIBRARIES)
      set(ASAN_RT_LIB_NAME libclang_rt.asan_osx_dynamic.dylib)
    else()
      set(INJECT_LIB_ENV_VAR LD_PRELOAD)
      if (${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
        set(ASAN_RT_LIB_NAME libclang_rt.asan-${HOST_ARCH}.so)
      elseif (${CMAKE_CXX_COMPILER_ID} MATCHES "GNU")
        set(ASAN_RT_LIB_NAME libasan.so)
      endif()
    endif()

    execute_process(
      COMMAND ${CMAKE_CXX_COMPILER} --print-file-name=${ASAN_RT_LIB_NAME}
      OUTPUT_VARIABLE ASAN_RT_LIB_PATH
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE PRINT_FILENAME_RESULT
    )
    if (PRINT_FILENAME_RESULT)
      message(FATAL_ERROR "Failed to get ASAN runtime library path.")
    endif()

    list(APPEND PYTHON_API_TESTS_ASAN_ENV "${INJECT_LIB_ENV_VAR}=${ASAN_RT_LIB_PATH}")
  endif()
endif()


# Add E2E tests for the Modular Python API.
add_test(NAME EndToEndTest
  COMMAND ${CMAKE_COMMAND} -E env
    ${PYTHON_API_TESTS_ASAN_ENV}
    LLVM_PROFILE_FILE=${CMAKE_BINARY_DIR}/profiles/EndToEndTest%4m.profraw
    ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/pytests
)

add_custom_target(check-EndToEndTest
  COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -R EndToEndTest
  DEPENDS "${GENERATED_TESTS_BASE}"
)

# TODO(#9138) modular-pybind-ext only exists if STOCK_TENSORFLOW_FOUND.
# Fix all relevant cmake files #8927 is resolved i.e. multi-tenancy lands.
if(STOCK_TENSORFLOW_FOUND AND ENABLE_END_TO_END_TEST)
  add_dependencies(check-EndToEndTest modular-pybind-ext)
  add_dependencies(check-genericml check-EndToEndTest)
endif()
