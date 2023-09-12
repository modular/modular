##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_test(NAME mojo-lsp-server-pytests
  COMMAND ${CMAKE_COMMAND} -E
    env ${ASAN_LIB_INJECT_ENV} MOJO_LSP_SERVER=$<TARGET_FILE:mojo-lsp-server>
    ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}
)
add_custom_target(check-mojo-lsp-server-pytests
  COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -R mojo-lsp-server-pytests
  DEPENDS
    "${GENERATED_TESTS_BASE}"
    mojo-lsp-server
    mojo-sdk
)

add_dependencies(check-mojo-lsp-server check-mojo-lsp-server-pytests)
