##===- CMakeLists.txt -----------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_subdirectory(include/Support/IndexDialect)
add_subdirectory(include/Support/IndexToLLVM)
#add_subdirectory(integration_test)
add_subdirectory(lib)
add_subdirectory(tools/index-opt)
add_subdirectory(test)
add_subdirectory(unittests)
