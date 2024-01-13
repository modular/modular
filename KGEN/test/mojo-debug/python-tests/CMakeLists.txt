##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_pytest(mojo-debug

    SKIP_PRELOAD_ASAN_RT

    DEPENDS
    "${GENERATED_TESTS_BASE}"
    mojo-sdk
    mojo

    ENV_VARS
    LLDB_TEST_INIT_FILE=${LLDB_TEST_INIT_FILE}

    EXTRA_ARGS
    -rs
    -v

    TOP_LEVEL_CHECK
    check-mojo-debug

    TEST_LOCATION
    ${CMAKE_CURRENT_SOURCE_DIR}
)
