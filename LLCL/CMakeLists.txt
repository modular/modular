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

# Because of the absence of the test directory, we need to add a custom target
# which mimics the target it would have created.
add_custom_target(check-llcl)
add_subdirectory(unittests)
