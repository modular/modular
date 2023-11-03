##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

# ASAN triggers errors because of python, so we skip.
if (LLVM_USE_SANITIZER)
  return()
endif()

add_test(NAME mojo-lldb-dwarf-pytests
  COMMAND ${CMAKE_COMMAND} -E env
    LLVM_PROFILE_FILE=${CMAKE_BINARY_DIR}/profiles/mojo-lldb-dwarf-pytests%4m.profraw
    LLDB_TEST_INIT_FILE=${LLDB_TEST_INIT_FILE}
    ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}

)
add_custom_target(check-mojo-lldb-dwarf-pytests
  COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -R mojo-lldb-dwarf-pytests
  DEPENDS
    "${GENERATED_TESTS_BASE}"
    mojo-sdk
    mojo
)

add_dependencies(check-mojo-lldb-dwarf check-mojo-lldb-dwarf-pytests)
