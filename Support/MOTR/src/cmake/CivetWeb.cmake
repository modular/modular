##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

# Civetweb configuration
include(FetchContent)
FetchContent_Declare(
    civetweb
    GIT_REPOSITORY https://github.com/civetweb/civetweb.git
    GIT_TAG 7f95a2632ef651402c15c39b72c4620382dd82bf
)

set(CIVETWEB_BUILD_TESTING OFF CACHE BOOL "Disable civetweb tests" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Disable shared libraries" FORCE)

# Civetweb C++ wrapper is disabled because it is not compatible with -fno-exceptions
set(CIVETWEB_ENABLE_CXX OFF CACHE BOOL "Enable C++ wrapper" FORCE)
set(CIVETWEB_BUILD_ENABLE_HTTP2 ON CACHE BOOL "Enable HTTP/2" FORCE)
set(CIVETWEB_ENABLE_IPV6 ON CACHE BOOL "Enable IPv6" FORCE)
set(CIVETWEB_ENABLE_SERVER_STATS ON CACHE BOOL "Enable server stats" FORCE)
set(CIVETWEB_ENABLE_ZLIB ON CACHE BOOL "Enable zlib" FORCE)
set(CIVETWEB_ENABLE_ASAN OFF CACHE BOOL "Enable ASAN" FORCE)
set(CIVETWEB_INSTALL_EXECUTABLE OFF CACHE BOOL "Disable installing civetweb executable" FORCE)
set(CIVETWEB_ENABLE_SSL OFF CACHE BOOL "Disable SSL" FORCE)
set(CIVETWEB_ENABLE_LUA OFF CACHE BOOL "Disable Lua" FORCE)
set(CIVETWEB_ENABLE_WEBSOCKETS ON CACHE BOOL "Enable websockets" FORCE)

FetchContent_MakeAvailable(civetweb)

# Add warning suppression for civetweb
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang")
    target_compile_options(civetweb-c-library PRIVATE
        -Wno-shorten-64-to-32
        -Wno-sign-conversion
        -Wno-declaration-after-statement
        -Wno-date-time
        -Wno-atomic-implicit-seq-cst
        -Wno-alloca
        -Wno-cast-align
        -Wno-extra-semi-stmt
        -Wno-unsafe-buffer-usage
        -Wno-switch-default
    )

    target_compile_options(civetweb-c-executable PRIVATE
        -Wno-missing-variable-declarations
        -Wno-date-time
        -Wno-sign-conversion
        -Wno-conversion
    )
    set_target_properties(civetweb-c-executable PROPERTIES
        EXCLUDE_FROM_ALL TRUE)
endif()