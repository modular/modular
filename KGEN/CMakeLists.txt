##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

set(KGEN_SOURCE_DIR ${MODULAR_SOURCE_DIR}/KGEN)
set(KGEN_BINARY_DIR ${MODULAR_BINARY_DIR}/KGEN)

find_package(Python3 COMPONENTS Interpreter REQUIRED)

# add_kgen_translation_library is a wrapper around add_modular_mlir_library that
# automatically links the library into kgen-translate.
function(add_kgen_translation_library name)
  set_property(GLOBAL APPEND PROPERTY KGEN_TRANSLATION_LIBS ${name})
  add_modular_mlir_library(${ARGV} DEPENDS modular-headers)
endfunction()

add_subdirectory(include/KGEN)
add_subdirectory(lib/HLCFDialect)
add_subdirectory(lib/KGENDialect)
add_subdirectory(lib/POPDialect)
add_subdirectory(lib/LITDialect)
add_subdirectory(lib/LowerLIT)
add_subdirectory(lib/KGENToLLVM)
add_subdirectory(lib/CompilerRT)
add_subdirectory(lib/MojoParser)
add_subdirectory(lib/Transforms)
add_subdirectory(lib/ExecutionEngine)
add_subdirectory(lib/CLOptions)
add_subdirectory(lib/Elaborator)
add_subdirectory(lib/LowerToObject)
add_subdirectory(lib/Compiler)

if(LLDB_FOUND)
  add_subdirectory(lib/MojoLLDB)
  add_subdirectory(lib/MojoJupyter)
endif()

add_subdirectory(tools/kgen)
add_subdirectory(tools/kgen-elaborate-opt)
add_subdirectory(tools/kgen-opt)
add_subdirectory(tools/kgen-translate)
add_subdirectory(tools/mojo-lsp-server)

if(LLDB_FOUND)
  add_subdirectory(tools/mojo-jupyter-executor)
  add_subdirectory(tools/mojo-repl-entry-point)
endif()

add_subdirectory(tools/mojo)
add_subdirectory(tools/mojo-driver)

# This function reads ${KGEN_LINK_LIBS} and puts the include flags into
# ${KGEN_INCLUDE_FLAGS}.
macro(_setup_libs)
  set(KGEN_LIB_DEPENDS "")
  set(KGEN_INCLUDE_FLAGS "")
  foreach (lib ${KGEN_LINK_LIBS})
    if (NOT TARGET ${lib})
      message(FATAL_ERROR "${lib} is not a target, it can't be linked")
    endif ()
    get_target_property(lib_include_dir ${lib} INCLUDE_DIRECTORIES)
    list(APPEND KGEN_INCLUDE_FLAGS "-I=${lib_include_dir}")
    list(APPEND KGEN_LIB_DEPENDS "${lib}")
  endforeach ()
endmacro()

# This function reads MODULAR_KERNELS_USE_SANITIZER and sets up the KGEN flags
# to enable sanitizers.
macro(_setup_santizer_flags)
  set(KGEN_SANITIZER_OPTIONS "")
  set(KGEN_SANITIZER_LIBS "")

  string(TOLOWER ${MODULAR_KERNELS_USE_SANITIZER} lower_MODULAR_KERNELS_USE_SANITIZER)
  if (lower_MODULAR_KERNELS_USE_SANITIZER STREQUAL "address")
    set(KGEN_SANITIZER_OPTIONS "--sanitize=address")
    set(KGEN_SANITIZER_LIBS ${KGEN_ASAN_LIBRARY})
  elseif (lower_MODULAR_KERNELS_USE_SANITIZER STREQUAL "thread")
    set(KGEN_SANITIZER_OPTIONS "--sanitize=thread")
    set(KGEN_SANITIZER_LIBS ${KGEN_TSAN_LIBRARY})
  endif()
endmacro()

# This function reads the MODULAR_KERNELS_BUILD_TYPE and sets up the KGEN
# optimization and debug flags into KGEN_OPTIMIZATION_LEVEL and KGEN_DEBUG_LEVEL
# respectively.
macro(_setup_debug_flags)
  string(TOLOWER ${MODULAR_KERNELS_BUILD_TYPE} build_type)
  set(KGEN_OPTIMIZATION_LEVEL 3)
  set(KGEN_DEBUG_LEVEL "none")
  if (build_type STREQUAL "debug")
    set(KGEN_OPTIMIZATION_LEVEL 0)
    set(KGEN_DEBUG_LEVEL "synthetic")
  elseif (build_type STREQUAL "relwithdebinfo")
    set(KGEN_DEBUG_LEVEL "synthetic")
  endif()

  # The caller specified no debug, so disable debugging.
  if (KGEN_NO_DEBUG)
    set(KGEN_DEBUG_LEVEL "none")
  endif()
endmacro()


# This function takes a target name, a single source file, and a list of
# targets to link. It generates the pre-elaboration library files to be used
# by other kgen files.
#
#  add_kgen_library(lib
#    OUTPUT library.mlirbc
#    source_file_1.lib.mlir
#    source_file_2.lib.mlir
#    LINK_LIBS someother-lib)
#
# This attaches the directory for the output to the target, so you can simply
# specify this target in the LINK_LIBS of another kgen_library or kgen_kernel
# and the include flags will be setup correctly.
function(add_kgen_library target)
  cmake_parse_arguments(KGEN "SAVE_TEMPS;NO_DEBUG" "OUTPUT" "LINK_LIBS" ${ARGN})

  _setup_libs()
  _setup_debug_flags()

  if (NOT KGEN_OUTPUT)
    set(KGEN_OUTPUT ${target}.lib.mlirbc)
  endif ()

  add_custom_command(
    OUTPUT ${KGEN_OUTPUT}
    COMMAND $<TARGET_FILE:kgen>
      $<$<BOOL:${KGEN_SAVE_TEMPS}>:-save-temps>
      -gen-lib ${KGEN_UNPARSED_ARGUMENTS} -o ${KGEN_OUTPUT}
      -O${KGEN_OPTIMIZATION_LEVEL}
      -debug-level=${KGEN_DEBUG_LEVEL}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS kgen ${KGEN_LIB_DEPENDS} ${KGEN_UNPARSED_ARGUMENTS} ${KGEN_LINK_LIBS}
  )

  add_custom_target(${target} DEPENDS ${KGEN_OUTPUT} ${KGEN_UNPARSED_ARGUMENTS} kgen ${KGEN_LIB_DEPENDS})
  set_target_properties(${target} PROPERTIES INCLUDE_DIRECTORIES ${CMAKE_CURRENT_BINARY_DIR})
endfunction()

# This function takes a target name, a single source file, and a list of
# kernels to be processed. It cannot execute kernels, but it will generate
# the CMake targets needed to depend on them from C/C++ code. Usage:
#
#  add_kgen_kernel(exp
#    SFF_DEBUG #< This prints information about the SelectFastestFunction
#              #< search process.
#
#    SOURCE exp.mlir
#    KERNELS run_exp
#    LINK_LIBS lib)
#
# This will generate two targets, `exp` and `exp-headers`. The former is
# the actual kernel, and the latter is a library that contains the headers
# for the kernel. Targets that depend on the headers should add `exp-headers`
# to their `DEPENDS` list.
#
# This function will pull its include flags from the linked libs to avoid
# having too many search paths.
function(add_kgen_kernel target)
  cmake_parse_arguments(KGEN "SFF_DEBUG;ENABLE_SEARCH;NO_DEBUG" "SOURCE;OUTPUT_NAME" "KERNELS;LINK_LIBS" ${ARGN})

  _setup_libs()
  _setup_debug_flags()

  if (MODULAR_KERNELS_USE_SANITIZER)
    _setup_santizer_flags()
  endif()

  # If we built with MODULAR_KGEN_ENABLE_SEARCH=ON, then turn on
  # KGEN_ENABLE_SEARCH unconditionally.
  if (MODULAR_KGEN_ENABLE_SEARCH)
    set(KGEN_ENABLE_SEARCH ON)
  endif ()

  if (NOT KGEN_OUTPUT_NAME)
    get_filename_component(KGEN_OUTPUT_NAME ${KGEN_SOURCE} NAME_WE)
  endif ()

  # Add the command used to generate the header for the kernel.
  set(header_file_name "${KGEN_OUTPUT_NAME}.h")
  file(RELATIVE_PATH header_file_name_rel ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/${header_file_name})
  add_custom_command(
    OUTPUT ${header_file_name}

    COMMAND $<TARGET_FILE:kgen> -emit-header
    -o ${header_file_name_rel} ${KGEN_INCLUDE_FLAGS} ${KGEN_SOURCE}
    -d ${header_file_name_rel}.d
    -O${KGEN_OPTIMIZATION_LEVEL}
    -debug-level=${KGEN_DEBUG_LEVEL}
    ${KGEN_SANITIZER_OPTIONS}
    $<$<BOOL:${MODULAR_ENABLE_XRAY}>:-xray-instrument>

    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    DEPFILE ${CMAKE_CURRENT_BINARY_DIR}/${header_file_name}.d
    DEPENDS kgen ${KGEN_SOURCE} ${KGEN_LIB_DEPENDS}
  )
  add_custom_target(${target}-headers DEPENDS ${header_file_name})
  add_dependencies(${target}-headers ${KGEN_LIB_DEPENDS} kgen)
  set_target_properties(${target}-headers PROPERTIES
    INCLUDE_DIRECTORIES ${CMAKE_CURRENT_BINARY_DIR}
  )

  # CMake emits build targets as relative paths but Ninja doesn't identify
  # absolute paths (in *.d) as relative paths (in build.ninja).
  set(archive_file_name "${KGEN_OUTPUT_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX}")
  file(RELATIVE_PATH archive_file_name_rel ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/${archive_file_name})

  # Add the command used to generate the archive for the kernel.
  add_custom_command(
    OUTPUT ${archive_file_name}

    COMMAND $<TARGET_FILE:kgen> -emit
    $<$<BOOL:${KGEN_SFF_DEBUG}>:-debug-only="select-fastest-function">
    $<$<BOOL:${KGEN_ENABLE_SEARCH}>:-enable-search>
    -o ${archive_file_name_rel} ${KGEN_INCLUDE_FLAGS} ${KGEN_SOURCE}
    -d ${archive_file_name_rel}.d
    -O${KGEN_OPTIMIZATION_LEVEL}
    -debug-level=${KGEN_DEBUG_LEVEL}
    ${KGEN_SANITIZER_OPTIONS}
    $<$<BOOL:${MODULAR_ENABLE_XRAY}>:-xray-instrument>

    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    DEPFILE ${CMAKE_CURRENT_BINARY_DIR}/${archive_file_name}.d
    # This depends on the header target because that will run (and therefore cache)
    # elaboration, thus saving much of the work for this target.
    DEPENDS kgen ${KGEN_SOURCE} ${KGEN_LIB_DEPENDS} ${target}-headers
  )
  add_custom_target(${target}-archive DEPENDS ${archive_file_name})

  add_library(${target} IMPORTED STATIC)
  add_dependencies(${target} ${KGEN_LIB_DEPENDS} kgen ${target}-archive)
  target_link_libraries(${target} INTERFACE KGENCompilerRT ${KGEN_SANITIZER_LIBS})
  set_target_properties(${target} PROPERTIES
    IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/${archive_file_name}
    LINKER_LANGUAGE C
    INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_BINARY_DIR}
    INCLUDE_DIRECTORIES ${CMAKE_CURRENT_BINARY_DIR}
  )
endfunction()

function(add_mojo_doc visibility doc_filename output_file output_directory)
  _setup_libs()

  set(GEN_DOC_JSON_FILE ${MODULAR_BINARY_DIR}/docs/${output_directory}${output_file}.mojodoc.json)
  set(GEN_DOC_FILE ${MODULAR_BINARY_DIR}/docs/${output_directory}${output_file}.md)

  # Generate the JSON file containing the documentation description.
  add_custom_command(
    OUTPUT "${GEN_DOC_JSON_FILE}"

    COMMAND $<TARGET_FILE:mojo-driver> doc
    -o ${GEN_DOC_JSON_FILE} ${KGEN_INCLUDE_FLAGS} ${doc_filename}
    -I ${KERNELS_BINARY_DIR}/mojo/Python

    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    DEPENDS mojo ${doc_filename} ${KGEN_LIB_DEPENDS}
  )

  # Process the JSON file into a markdown that can be rendered.
  add_custom_command(
    OUTPUT "${GEN_DOC_FILE}"

    DEPENDS
      ${MODULAR_SOURCE_DIR}/KGEN/utils/mojodoc_json_to_markdown.py
      "${GEN_DOC_JSON_FILE}"

    COMMAND
      "${Python3_EXECUTABLE}"
      ${MODULAR_SOURCE_DIR}/KGEN/utils/mojodoc_json_to_markdown.py
      -o ${GEN_DOC_FILE} ${GEN_DOC_JSON_FILE}

    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )

  add_custom_target(${output_file}DocGen DEPENDS ${GEN_DOC_FILE})
  set(is_public FALSE)
  if("${visibility}" STREQUAL "PUBLIC")
    set(is_public TRUE)
  endif()

  if (is_public)
    add_dependencies(modular-doc ${output_file}DocGen)
  else()
    add_dependencies(modular-internal-doc ${output_file}DocGen)
  endif()

  # Add a custom target that does the following:
  # 1. Invokes `mojo -doc-validate` on the given Mojo file, writing any and all
  #    output to a temporary file.
  # 2. Invokes a CMake script that searches that output file for `-doc-validate`
  #    warnings originating from the given Mojo file.
  #
  # This is a hack to work around the fact that `mojo -doc-validate foo.mojo`
  # not only validates the documentation for `foo.mojo`, but also all of its
  # transitive imports. Here, the custom target `fooDocValidate` will only fail
  # if a `-doc-validate` warning is produced from within `foo.mojo`, not any of
  # its imported files.

  # Invoke `mojo -doc-validate` and pipe all output to a file ending in
  # `warnings.txt`.
  set(DOC_VALIDATE_WARNINGS_FILE ${GEN_DOC_JSON_FILE}.warnings.txt)
  add_custom_command(
    OUTPUT "${DOC_VALIDATE_WARNINGS_FILE}"

    COMMAND $<TARGET_FILE:mojo-driver> doc -validate
    -o /dev/null ${KGEN_INCLUDE_FLAGS} ${doc_filename}
    -I ${KERNELS_BINARY_DIR}/mojo/Python
    > "${DOC_VALIDATE_WARNINGS_FILE}" 2>&1

    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    DEPENDS mojo ${doc_filename} ${KGEN_LIB_DEPENDS}
  )

  # Add a `DocValidate` target that checks the `-doc-validate` output for
  # warnings output from within the original Mojo file.
  get_filename_component(doc_filename_we ${doc_filename} NAME_WE)
  string(CONCAT validate_comment
    "Scanning \"${DOC_VALIDATE_WARNINGS_FILE}\" for warnings originating from "
    "\"${doc_filename}\".")
  string(CONCAT failure_message
    "The following warnings were output when validating the doc strings in "
    "\"${doc_filename}\":")
  add_custom_target("${doc_filename_we}DocValidate"
    COMMENT "${validate_comment}"

    COMMAND
      ${CMAKE_COMMAND} -DPATH="${DOC_VALIDATE_WARNINGS_FILE}"
      -DPATTERN="${doc_filename}:[0-9]*:[0-9]*: warning:[^\\^]*\\^"
      -DFAILURE_MESSAGE="${failure_message}"
      -P ${MODULAR_SOURCE_DIR}/KGEN/CheckPatternInFile.cmake

    DEPENDS "${DOC_VALIDATE_WARNINGS_FILE}"
  )

  # Only add PUBLIC Mojo files to the top-level doc validation target.
  # This ensures that `modular-doc-validate` only fails when our public
  # libraries are legitimately missing documentation.
  if (is_public)
    add_dependencies(modular-doc-validate "${doc_filename_we}DocValidate")
  endif()
endfunction()

# Testing infrastructure
add_subdirectory(test)
add_subdirectory(unittests)
#add_subdirectory(integration_test)
