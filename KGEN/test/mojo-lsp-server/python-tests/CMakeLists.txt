##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

# ASAN triggers errors because of python, so we skip.
if (LLVM_USE_SANITIZER)
  return()
endif()

add_test(NAME mojo-lsp-server-pytests
  COMMAND ${CMAKE_COMMAND} -E env
    MOJO_LSP_SERVER=$<TARGET_FILE:mojo-lsp-server>
    LLVM_PROFILE_FILE=${CMAKE_BINARY_DIR}/profiles/mojo-lsp-server-pytests%4m.profraw
    ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR} -v --junitxml=junit/test-suite-mojo-lsp-server-pytests-test-results.xml
)

add_custom_target(check-mojo-lsp-server-pytests
  COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -R mojo-lsp-server-pytests
  DEPENDS
    "${GENERATED_TESTS_BASE}"
    mojo-lsp-server
    mojo-sdk
)

add_dependencies(check-mojo-lsp-server check-mojo-lsp-server-pytests)
