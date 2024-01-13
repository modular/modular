##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_pytest(mojo-lsp-server

  SKIP_PRELOAD_ASAN_RT

  ENV_VARS
  MOJO_LSP_SERVER=$<TARGET_FILE:mojo-lsp-server>

  TEST_LOCATION
  ${CMAKE_CURRENT_SOURCE_DIR}

  DEPENDS
  "${GENERATED_TESTS_BASE}"
  mojo-lsp-server
  mojo-sdk

  TOP_LEVEL_CHECK
  check-mojo-lsp-server
)
