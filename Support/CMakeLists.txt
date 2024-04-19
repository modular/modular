##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_subdirectory(include/Support/DebugInfoDialect/DebugInfoToLLVM)
add_subdirectory(include/Support/DebugInfoDialect/IR)
add_subdirectory(include/Support/DebugInfoDialect/Transforms)
add_subdirectory(include/Support/MDialect)
add_subdirectory(lib)
add_subdirectory(tools/build-info)
add_subdirectory(tools/crash-test-dummy EXCLUDE_FROM_ALL)
add_subdirectory(tools/driver-tblgen)
add_subdirectory(tools/compare-timings)
add_subdirectory(examples/greeter-cli)
