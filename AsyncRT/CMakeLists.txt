##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

set(ASYNCRT_SOURCE_DIR ${MODULAR_SOURCE_DIR}/AsyncRT)
set(ASYNCRT_BINARY_DIR ${MODULAR_BINARY_DIR}/AsyncRT)

# Add ASyncRT files to the include path
set(ASYNCRT_INCLUDE_DIR ${ASYNCRT_SOURCE_DIR}/include)
include_directories(${ASYNCRT_INCLUDE_DIR})
set(ASYNCRT_INCLUDE_DIR ${ASYNCRT_BINARY_DIR}/include)
include_directories(${ASYNCRT_INCLUDE_DIR})

add_subdirectory(lib/Support)
add_subdirectory(lib/Runtime)
add_subdirectory(lib/CompilerSupport)
add_subdirectory(lib/Init)

add_subdirectory(tools/crash-report-path-info EXCLUDE_FROM_ALL)
add_subdirectory(tools/crash-test-dummy EXCLUDE_FROM_ALL)

add_subdirectory(unittests)
add_subdirectory(benchmarks)
add_subdirectory(test)
