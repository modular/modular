##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_subdirectory(include/Cache/CacheDialect)
add_subdirectory(include/Cache/CachePasses)
add_subdirectory(lib)
add_subdirectory(tools/cache-opt)
add_subdirectory(tools/cache-mgr)
add_subdirectory(test)
add_subdirectory(unittests)

# This function allows us to cache build artifacts from within cmake, and
# generate header files that contain those CAS IDs. Use like this (example
# stolen from KGEN/lib/ExecutionEngine):
#
# cache_files(cache_a_file
#             BASE_DIR ${MODULAR_DERIVED_PATH}/.kgen_cache/orc
#             HEADER_NAME ${MODULAR_BINARY_DIR}/KGEN/include/KGEN/ExecutionEngine/ORCCASID.h
#             INPUTS ${PATH_TO_ORC_RT}
#             INPUT_NAMES "MODULAR_ORC_RT_CAS_ID")
function(cache_files target)
  cmake_parse_arguments(CACHE_MGR "" "BASE_DIR;HEADER_NAME" "INPUTS;INPUT_NAMES" ${ARGN})
  # Set the variables the script is gonna need.
  set(FILENAME ${CMAKE_CURRENT_BINARY_DIR}/hashes.out)
  set(OUTFILE ${CACHE_MGR_HEADER_NAME})
  set(NAMES ${CACHE_MGR_INPUT_NAMES})
  configure_file(${CMAKE_SOURCE_DIR}/Cache/cmake/ReadHashes.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/ReadHashes.cmake @ONLY)
  # Add a target to run that script to generate the header file.
  add_custom_target(${target} ALL
    COMMAND $<TARGET_FILE:cache-mgr> -base-dir=${CACHE_MGR_BASE_DIR} -input=${CACHE_MGR_INPUTS} -o=${CMAKE_CURRENT_BINARY_DIR}/hashes.out
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/ReadHashes.cmake)
endfunction()
