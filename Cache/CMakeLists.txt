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
#             BASE_DIR ${MODULAR_DERIVED_PATH}/.mojo_cache/orc
#             HEADER_NAME ${MODULAR_BINARY_DIR}/KGEN/include/KGEN/ExecutionEngine/ORCCASID.h
#             INPUT_NAMES "kOrcRT" # Note that these must be globally unique - they're turned into C++ identifiers.
#             RELATIVE_INSTALL_DIR ".mojo_cache/orc")
function(cache_files target)
  cmake_parse_arguments(CACHE_MGR "" "BASE_DIR;HEADER_NAME;VERSION" "INPUTS;INPUT_NAMES;RELATIVE_INSTALL_DIR" ${ARGN})

  # For debugging, but default to always for now.
  # This is the name of the directory used in `.derived/.mojo_cache`
  # for where to store the relevant CAS artifacts.  Before filing a bug
  # report, please check the contents there.
  include(CMakePrintHelpers)
  cmake_print_variables(CACHE_MGR_VERSION)

  # Input names must all be unique, so we can use the first one.
  list(GET CACHE_MGR_INPUT_NAMES 0 UNIQUE_PATH)
  set(UNIQUE_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/${UNIQUE_PATH})
  # Set the variables the script is gonna need.
  set(FILENAME ${UNIQUE_BINARY_DIR}/hashes.out)
  # rm the file and create it anew to ensure it's empty.
  add_custom_target(${target}_clean
    COMMAND ${CMAKE_COMMAND} -E rm -f ${FILENAME}
    COMMAND ${CMAKE_COMMAND} -E touch ${FILENAME})

  # Add a target to run that script to generate the header file.
  add_custom_target(${target}
    COMMAND ${CMAKE_COMMAND} -P ${UNIQUE_BINARY_DIR}/ReadHashes.cmake
    DEPENDS ${UNIQUE_BINARY_DIR}/ReadHashes.cmake ${target}_clean)

  list(LENGTH CACHE_MGR_INPUTS num_inputs)
  if (num_inputs EQUAL 1)
    set(CACHE_MGR_INPUTS "${CACHE_MGR_INPUTS};")
    set(CACHE_MGR_INPUT_NAMES "${CACHE_MGR_INPUT_NAMES};")
  endif ()
  # CMake foreach loops are inclusive, so we have to iterate up to num_inputs - 1
  math(EXPR iters "${num_inputs} - 1")

  foreach (idx RANGE ${iters})
    list(GET CACHE_MGR_INPUTS ${idx} input)
    list(GET CACHE_MGR_INPUT_NAMES ${idx} name)
    add_custom_target(${target}_${name} ALL
      COMMAND $<TARGET_FILE:cache-mgr> -base-dir=${CACHE_MGR_BASE_DIR} -input=${input} -backend-version=${CACHE_MGR_VERSION} -o=${FILENAME} -append
      # If an input dependency that would change the KGEN version changes,
      # we need to have the `cache-mgr` put the new CAS ID for the artifact in.
      # Otherwise, you will have stale CAS IDs and may run into runtime issues
      # in the KGEN ExecutionEngine when looking up the CAS ID, such as for
      # `compiler-rt-cas`.
      DEPENDS ${target}_clean KGENVersion)

    add_dependencies(${target} ${target}_${name})
  endforeach ()

  set(OUTFILE ${CACHE_MGR_HEADER_NAME})
  set(TMPFILE "${CACHE_MGR_HEADER_NAME}.tmp")
  set(NAMES ${CACHE_MGR_INPUT_NAMES})
  configure_file(${CMAKE_SOURCE_DIR}/Cache/cmake/ReadHashes.cmake.in ${UNIQUE_BINARY_DIR}/ReadHashes.cmake @ONLY)

  install(DIRECTORY "${CACHE_MGR_BASE_DIR}/"
    DESTINATION ${CACHE_MGR_RELATIVE_INSTALL_DIR})
endfunction()
