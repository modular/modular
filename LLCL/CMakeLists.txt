#===- CMakeLists.txt ------------------------------------------*- cmake -*-===##
#
# This file is Modular Inc proprietary.
#
#===-----------------------------------------------------------------------===##

set(LLCL_SOURCE_DIR ${MODULAR_SOURCE_DIR}/LLCL)
set(LLCL_BINARY_DIR ${MODULAR_BINARY_DIR}/LLCL)

# Add LLCL files to the include path
set(LLCL_INCLUDE_DIR ${LLCL_SOURCE_DIR}/include)
include_directories(${LLCL_INCLUDE_DIR})
set(LLCL_INCLUDE_DIR ${LLCL_BINARY_DIR}/include)
include_directories(${LLCL_INCLUDE_DIR})

#add_subdirectory(include/LLCL)
#add_subdirectory(integration_test)
add_subdirectory(lib/Support)
add_subdirectory(lib/Runtime)
#add_subdirectory(test)
#add_subdirectory(tools)
#add_subdirectory(unittests)
