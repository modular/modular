##===- CMakeLists.txt -----------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_custom_target(check-support)
add_dependencies(check-all check-support)

function(add_msupport_test target)
  add_modular_test(${target} ${ARGN}
    NAME ${target}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    TOP_LEVEL_CHECK check-support
    LINK_LIBS PRIVATE
    MSupport
    )
endfunction()

add_msupport_test(HMACTest HMACTest.cpp)
add_msupport_test(BlobCacheTest BlobCacheTest.cpp)

