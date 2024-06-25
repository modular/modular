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
add_subdirectory(tools/driver-tblgen)
add_subdirectory(tools/compare-timings)
add_subdirectory(tools/support-dialect-opt)
add_subdirectory(tools/system-info)
add_subdirectory(examples/greeter-cli)
add_subdirectory(test)
add_subdirectory(unittests)
