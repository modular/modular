##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

set(LLCL_SOURCE_DIR ${MODULAR_SOURCE_DIR}/LLCL)
set(LLCL_BINARY_DIR ${MODULAR_BINARY_DIR}/LLCL)

# Add LLCL files to the include path
set(LLCL_INCLUDE_DIR ${LLCL_SOURCE_DIR}/include)
include_directories(${LLCL_INCLUDE_DIR})
set(LLCL_INCLUDE_DIR ${LLCL_BINARY_DIR}/include)
include_directories(${LLCL_INCLUDE_DIR})

add_subdirectory(lib/Support)
add_subdirectory(lib/Runtime)
add_subdirectory(lib/CompilerSupport)
add_subdirectory(lib/Init)

add_subdirectory(tools/crash-report-path-info EXCLUDE_FROM_ALL)
add_subdirectory(tools/crash-test-dummy EXCLUDE_FROM_ALL)

add_subdirectory(unittests)
add_subdirectory(benchmarks)
add_subdirectory(test)
