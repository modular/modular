##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

cmake_minimum_required(VERSION 3.10)
project(motr)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
set(CMAKE_CXX_STANDARD 17)

add_compile_options(-fno-exceptions)
add_compile_options(-fno-rtti)

# CivetWeb.cmake fetches thirdparty CivetWeb
# source and provides civetweb-c-library target
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/CivetWeb.cmake)

# MOTR_VERSION.cmake parses version.txt and provides set_motr_version_defines
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/MOTR_VERSION.cmake)

add_library(motr-core STATIC
    Config/ConfigFile.cpp
    Watch/WatchDir.cpp
    Webserver/WebServer.cpp
    HostInfo.cpp
    ../common/MString.cpp
    ../common/StringLibrary.cpp
    ../common/TagLibrary.cpp
)

target_link_libraries(motr-core PUBLIC
    civetweb-c-library
)

if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    target_link_libraries(motr-core PUBLIC
        "-framework CoreServices"
    )
endif()

# Get absolute path to the MOTR third-party directory
get_filename_component(THIRD_PARTY_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../../third-party/MOTR"
  ABSOLUTE
)
message(STATUS "Third party directory: ${THIRD_PARTY_DIR}")


target_include_directories(motr-core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/include
    ${THIRD_PARTY_DIR}/json/include
    ${THIRD_PARTY_DIR}/rapidyaml/include
    ${THIRD_PARTY_DIR}/fmt/include)

set_motr_version_defines(motr-core)

target_compile_definitions(motr-core PUBLIC
    MOTR_ENABLED=1
)

add_executable(motr
    main.cpp
    status_main.cpp
    clean_main.cpp
    flags_main.cpp
    test_main.cpp
    server_main.cpp
    strings_main.cpp
    tags_main.cpp
)

target_link_libraries(motr PRIVATE
    motr-core
)
