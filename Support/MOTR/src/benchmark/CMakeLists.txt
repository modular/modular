##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

cmake_minimum_required(VERSION 3.10)
project(motr-benchmark)

# Get absolute path to third-party directory
get_filename_component(THIRD_PARTY_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../third-party"
    ABSOLUTE
)

message(STATUS "Third party directory: ${THIRD_PARTY_DIR}")

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
set(CMAKE_CXX_STANDARD 17)

add_compile_options(-fno-exceptions)
add_compile_options(-fno-rtti)

# Civetweb configuration
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Disable shared libraries" FORCE)


message(STATUS "--------------------------------")

add_executable(motr-benchmark
    main.cpp
)

target_include_directories(motr-benchmark PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/include
    ${THIRD_PARTY_DIR}/fmt/include
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON FORCE)

