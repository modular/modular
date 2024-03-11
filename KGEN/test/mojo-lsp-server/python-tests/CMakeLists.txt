##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_pytest(mojo-lsp-server
  SKIP_PRELOAD_ASAN_RT

  ENV_VARS
  MOJO_LSP_SERVER=$<TARGET_FILE:mojo-lsp-server>
  DISABLE_LSP_CRASH_RECOVERY=true

  TEST_LOCATION
  ${CMAKE_CURRENT_SOURCE_DIR}

  DEPENDS
  "${GENERATED_TESTS_BASE}"
  mojo-lsp-server
  mojo-sdk

  # Disabled due to https://github.com/modularml/modular/issues/34132
  # TOP_LEVEL_CHECK
  # check-mojo-lsp-server
  EXCLUDE_FROM_CHECK_ALL
)
