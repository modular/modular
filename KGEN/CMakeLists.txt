##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

set(KGEN_SOURCE_DIR ${MODULAR_SOURCE_DIR}/KGEN)
set(KGEN_BINARY_DIR ${MODULAR_BINARY_DIR}/KGEN)

find_package(Python3 COMPONENTS Interpreter REQUIRED)

# On MSVC compile with unicode sources so that we display 🔥 correctly.
if (MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /utf-8")
endif ()

# This target represents the full mojo-stdlib which contains all the standard
# library packages.
add_custom_target(mojo-stdlib)

# This target represents the full mojo-sdk, and builds both `mojo` and all of
# the necessary standard library packages.
add_custom_target(mojo-sdk DEPENDS mojo-stdlib)

# KGEN MLIR dialects.
add_subdirectory(include/KGEN)
add_subdirectory(lib/Support)
add_subdirectory(lib/Interpreter)
add_subdirectory(lib/HLCFDialect)
add_subdirectory(lib/KGENDialect)
add_subdirectory(lib/POPDialect)
add_subdirectory(lib/LITDialect)

# KGEN MLIR passes.
add_subdirectory(lib/LowerLIT)
add_subdirectory(lib/KGENToLLVM)
add_subdirectory(lib/Transforms)
add_subdirectory(lib/Elaborator)

# KGEN->KGEN MOGG specific transforms.
add_subdirectory(lib/MOGGPreElab)

# KGEN compiler and JIT.
add_subdirectory(lib/Pipeline)
add_subdirectory(lib/ToolCommon)
add_subdirectory(lib/CompilerRT)
add_subdirectory(lib/ExecutionEngine)
add_subdirectory(lib/ObjectCompiler)
add_subdirectory(lib/Compiler)
add_subdirectory(lib/Package)

# Mojo frontend.
add_subdirectory(lib/MojoParser)
add_subdirectory(lib/MojoTooling)

add_subdirectory(lib/MojoLLDB)
add_subdirectory(lib/MojoJupyter)

add_subdirectory(tools/hash-mlir)
add_subdirectory(tools/kgen)
add_subdirectory(tools/kgen-opt)
add_subdirectory(tools/kgen-translate)
add_subdirectory(tools/mojo-lsp-server)

add_subdirectory(tools/mojo-jupyter-executor)
add_subdirectory(tools/mojo-repl-entry-point)

add_subdirectory(tools/mojo)

# This function reads ${KGEN_LINK_LIBS} and puts the include flags into
# ${KGEN_INCLUDE_FLAGS}. It also read ${KGEN_LINK_DIRS} and appends those
# to the include path.
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
  foreach (lib_include_dir ${KGEN_LINK_DIRS})
    list(APPEND KGEN_INCLUDE_FLAGS "-I=${lib_include_dir}")
  endforeach ()
endmacro()

# This function reads MODULAR_KERNELS_USE_SANITIZER and sets up the KGEN flags
# to enable sanitizers.
macro(_setup_santizer_flags)
  if (MODULAR_KERNELS_USE_SANITIZER)
    set(KGEN_SANITIZER_OPTIONS "")
    set(KGEN_SANITIZER_LIBS "")

    string(TOLOWER ${MODULAR_KERNELS_USE_SANITIZER} lower_MODULAR_KERNELS_USE_SANITIZER)
    if (lower_MODULAR_KERNELS_USE_SANITIZER STREQUAL "address")
      set(KGEN_SANITIZER_OPTIONS "--sanitize" "address")
      set(KGEN_SANITIZER_LIBS ${KGEN_ASAN_LIBRARY})
    elseif (lower_MODULAR_KERNELS_USE_SANITIZER STREQUAL "thread")
      set(KGEN_SANITIZER_OPTIONS "--sanitize" "thread")
      set(KGEN_SANITIZER_LIBS ${KGEN_TSAN_LIBRARY})
    endif()
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
    set(KGEN_DEBUG_LEVEL "full")
  elseif (build_type STREQUAL "relwithdebinfo")
    set(KGEN_DEBUG_LEVEL "full")
  endif()

  # The caller specified no debug, so disable debugging.
  if (KGEN_NO_DEBUG)
    set(KGEN_DEBUG_LEVEL "none")
  endif()
endmacro()


# This function reads MODULAR_USE_NATIVE_ARCH / BUILD_ARCH and respects it to
# pass in the appropriate flags to the kgen compiler.
macro(_setup_arch)
  if (NOT CMAKE_APPLE_SILICON_PROCESSOR AND NOT "${CPU_BUILD_ARCH}" STREQUAL "")
    if (CPU_BUILD_ARCH STREQUAL "neoverse-n1")
      set(MOJO_ARCH_FLAGS "-mcpu" "neoverse-n1")
    elseif(MOJO_BUILD_ARCH STREQUAL "arm64")
      set(MOJO_ARCH_FLAGS "-march" "arm64")
    else()
      set(MOJO_ARCH_FLAGS "-march" "${CPU_BUILD_ARCH}")
    endif()
  endif()
endmacro()

# This function reads the cmake configuration and constructs the `-D` flags it
# needs to pass in to Mojo and KGEN.
macro(_setup_defines)
  string(TOLOWER ${CMAKE_BUILD_TYPE} build_type)
  string(TOLOWER ${MODULAR_KERNELS_BUILD_TYPE} kernels_build_type)
  set(MOJO_DEFINES)
  if (MODULAR_PARANOID)
    list(APPEND MOJO_DEFINES "-D" "MODULAR_PARANOID")
  endif()
  if (MODULAR_RELEASE_PACKAGE_BUILD)
    list(APPEND MOJO_DEFINES "-D" "MODULAR_RELEASE_PACKAGE_BUILD")
  endif()
  list(APPEND MOJO_DEFINES
    "-D" "MODULAR_LLCL_MAX_PROFILING_LEVEL=${MODULAR_LLCL_MAX_PROFILING_LEVEL}")
  list(APPEND MOJO_DEFINES
    "-D" "BUILD_TYPE=${build_type}")
  list(APPEND MOJO_DEFINES
    "-D" "KERNELS_BUILD_TYPE=${kernels_build_type}")
endmacro()

# Add the necessary tool dependencies to the mojo-sdk target.
add_dependencies(mojo-sdk mojo mojo-lsp-server MojoJupyter)

# This function will generate the CMake targets needed to build a Mojo package
# for the current source directory, or directory specified by PACKAGE_DIR.
#   Usage:
#
#  add_mojo_package(MojoPackage PUBLIC
#     Module
#     NestedPackage/Module
#  )
#
# This will generate a target, `MojoPackage`, and also generate documentation.
# Source files of the package are explicitly added. By default this will
# generate docs inside <build-dir>/docs/Kernels/docs. If DOCS_DIR is given the
# docs will instead be generated in <build-dir>/docs/${DOCS_DIR}.
# The visibility of the target can be specified as PUBLIC or PRIVATE, which
# controls whether the documentation is generated as part of the public API
# documentation.
#
# This function will pull its include flags from the linked libs to avoid
# having too many search paths.
function(add_mojo_package target visibility)
  cmake_parse_arguments(KGEN
                        "STDLIB"
                        "PACKAGE_DIR;OUTPUT_NAME;DOCS_DIR"
                        "LINK_LIBS;LINK_DIRS" ${ARGN})

  _setup_libs()
  _setup_defines()
  _setup_debug_flags()
  _setup_santizer_flags()

  if (NOT KGEN_PACKAGE_DIR)
    set(KGEN_PACKAGE_DIR ${CMAKE_CURRENT_LIST_DIR})
  endif ()
  if (NOT KGEN_OUTPUT_NAME)
    get_filename_component(KGEN_OUTPUT_NAME ${KGEN_PACKAGE_DIR} NAME_WE)
  endif ()
  if(NOT KGEN_DOCS_DIR)
    set(KGEN_DOCS_DIR "Kernels/docs")
  endif()

  # CMake emits build targets as relative paths but Ninja doesn't identify
  # absolute paths (in *.d) as relative paths (in build.ninja).
  set(package_file_name "${KGEN_OUTPUT_NAME}.mojopkg")
  file(RELATIVE_PATH package_file_name_rel ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/${package_file_name})

  set(KGEN_SOURCE_FILES "")
  foreach(file ${KGEN_UNPARSED_ARGUMENTS})
    list(APPEND KGEN_SOURCE_FILES "${KGEN_PACKAGE_DIR}/${file}.mojo")
  endforeach()

  # Ensure at least one source file is specified
  if (NOT KGEN_SOURCE_FILES)
    message(FATAL_ERROR "Must specify one or more mojo source files when calling `add_mojo_package`.")
  endif()

  # `mojo` doesn't expose synthetic debug info.
  if (${KGEN_DEBUG_LEVEL} STREQUAL "synthetic")
    set(KGEN_DEBUG_LEVEL "none")
  endif()

  # `mojo` is O3 by default, we only need to specify the flag if we're not
  # optimizing.
  set(KGEN_OPTIMIZATION_FLAG, "")
  if (${KGEN_OPTIMIZATION_LEVEL} EQUAL 0)
    set(KGEN_OPTIMIZATION_FLAG "-no-optimization")
  endif()

  # If this is not a standard library package, it should depend on the full-sdk.
  if (NOT KGEN_STDLIB)
    list(APPEND KGEN_LIB_DEPENDS mojo-sdk)
  endif()

  set(KGEN_PACKAGE_FILENAME "${KGEN_PACKAGE_DIR}/__init__.mojo")

  # Add the command used to generate the archive for the kernel.
  add_custom_command(
    OUTPUT ${package_file_name}

    COMMAND $<TARGET_FILE:mojo> package
    ${MOJO_DEFINES}
    ${KGEN_PACKAGE_DIR} -o ${package_file_name_rel} ${KGEN_INCLUDE_FLAGS}
    ${KGEN_OPTIMIZATION_FLAG}
    ${KGEN_SANITIZER_OPTIONS}
    --debug-level ${KGEN_DEBUG_LEVEL}

    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    DEPENDS mojo ${KGEN_SOURCE_FILES} ${KGEN_LIB_DEPENDS} ${KGEN_PACKAGE_FILENAME}
  )
  add_custom_target(${target} DEPENDS ${package_file_name})
  set_target_properties(${target} PROPERTIES INCLUDE_DIRECTORIES ${CMAKE_CURRENT_BINARY_DIR})

  # If this is a standard library package, then we need to add it to the
  # mojo-sdk and mojo-stdlib target.
  if (KGEN_STDLIB)
    add_dependencies(mojo-stdlib ${target})
    install(FILES "${CMAKE_BINARY_DIR}/${package_file_name_rel}"
            DESTINATION "lib/mojo"
            COMPONENT mojo-sdk)

  # Generate documentation for the package. Standard library documentation is
  # generated elsewhere.
  else()
    add_mojo_package_doc(${target}-doc ${visibility} ${KGEN_PACKAGE_DIR}
      ${KGEN_DOCS_DIR}/ DEPENDS ${KGEN_SOURCE_FILES})
  endif()
endfunction()


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
  cmake_parse_arguments(KGEN
    "DISABLE_PREBUILT_PACKAGES;SAVE_TEMPS;NO_DEBUG"
    "OUTPUT"
    "LINK_LIBS;LINK_DIRS"
    ${ARGN})

  _setup_libs()
  _setup_arch()
  _setup_defines()
  _setup_debug_flags()
  _setup_santizer_flags()

  if (NOT KGEN_OUTPUT)
    set(KGEN_OUTPUT ${target}.lib.mlirbc)
  endif ()
  # Workaround for us setting cmake_policy CMP0116 to OLD.  Ninja expects all
  # paths in depfiles to be relative paths from the build directory.  CMake
  # automatically fixes this if CMP0116 is NEW, but CMP0116 NEW is currently
  # incompatible with the LLVM TableGen rules, so we need to manually adjust
  # the output paths and working directory of custom commands that emit
  # DEPFILEs until then.
  file(RELATIVE_PATH kgen_output_rel ${CMAKE_BINARY_DIR} ${KGEN_OUTPUT})

  add_custom_command(
    OUTPUT ${KGEN_OUTPUT}
    COMMAND $<TARGET_FILE:kgen>
      ${MOJO_ARCH_FLAGS}
      ${MOJO_DEFINES}
      $<$<BOOL:${KGEN_DISABLE_PREBUILT_PACKAGES}>:-disable-prebuilt-packages>
      $<$<BOOL:${KGEN_SAVE_TEMPS}>:-save-temps>
      -gen-lib ${KGEN_UNPARSED_ARGUMENTS} -o ${kgen_output_rel}
      -O${KGEN_OPTIMIZATION_LEVEL}
      ${KGEN_SANITIZER_OPTIONS}
      -debug-level=${KGEN_DEBUG_LEVEL}
      -d ${CMAKE_CURRENT_BINARY_DIR}/${target}.d
    DEPFILE ${CMAKE_CURRENT_BINARY_DIR}/${target}.d
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    DEPENDS kgen
            mojo-sdk
            ${KGEN_LIB_DEPENDS}
            ${KGEN_LINK_LIBS}
            ${KGEN_UNPARSED_ARGUMENTS}
  )

  add_custom_target(${target} DEPENDS kgen
                                      mojo-stdlib
                                      ${KGEN_LIB_DEPENDS}
                                      ${KGEN_OUTPUT}
                                      ${KGEN_UNPARSED_ARGUMENTS})
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
  cmake_parse_arguments(KGEN
                        "SFF_DEBUG;NO_DEBUG"
                        "SOURCE;OUTPUT_NAME"
                        "KERNELS;DEPENDS;LINK_LIBS;LINK_DIRS" ${ARGN})

  _setup_libs()
  _setup_arch()
  _setup_defines()
  _setup_debug_flags()
  _setup_santizer_flags()

  if (NOT KGEN_OUTPUT_NAME)
    get_filename_component(KGEN_OUTPUT_NAME ${KGEN_SOURCE} NAME_WE)
  endif ()

  # Add the command used to generate the header for the kernel.
  set(header_file_name "${KGEN_OUTPUT_NAME}.h")
  file(RELATIVE_PATH header_file_name_rel ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/${header_file_name})
  add_custom_command(
    OUTPUT ${header_file_name}

    COMMAND $<TARGET_FILE:kgen> -emit-header
    ${MOJO_ARCH_FLAGS}
    ${MOJO_DEFINES}
    -enable-search
    -o ${header_file_name_rel} ${KGEN_INCLUDE_FLAGS} ${KGEN_SOURCE}
    -d ${header_file_name_rel}.d
    -O${KGEN_OPTIMIZATION_LEVEL}
    -debug-level=${KGEN_DEBUG_LEVEL}
    ${KGEN_SANITIZER_OPTIONS}
    $<$<BOOL:${MODULAR_ENABLE_XRAY}>:-xray-instrument>

    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    DEPFILE ${CMAKE_CURRENT_BINARY_DIR}/${header_file_name}.d
    DEPENDS kgen ${KGEN_SOURCE} ${KGEN_LIB_DEPENDS} mojo-sdk
  )
  add_custom_target(${target}-headers DEPENDS ${header_file_name})
  add_dependencies(${target}-headers ${KGEN_LIB_DEPENDS} kgen mojo-sdk ${KGEN_DEPENDS})
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
    ${MOJO_ARCH_FLAGS}
    ${MOJO_DEFINES}
    $<$<BOOL:${KGEN_SFF_DEBUG}>:-debug-only="select-fastest-function">
    -enable-search
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
    DEPENDS kgen ${KGEN_SOURCE} ${KGEN_LIB_DEPENDS} ${target}-headers mojo-sdk
  )
  add_custom_target(${target}-archive DEPENDS ${archive_file_name})

  add_library(${target} IMPORTED STATIC)
  add_dependencies(${target} ${KGEN_LIB_DEPENDS} kgen mojo-sdk ${target}-archive)
  target_link_libraries(${target} INTERFACE KGENCompilerRT)
  if (NOT LLVM_USE_SANITIZER)
    target_link_libraries(${target} INTERFACE KGENCompilerRT ${KGEN_SANITIZER_LIBS})
  endif()
  set_target_properties(${target} PROPERTIES
    IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/${archive_file_name}
    LINKER_LANGUAGE C
    INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_BINARY_DIR}
    INCLUDE_DIRECTORIES ${CMAKE_CURRENT_BINARY_DIR}
  )
endfunction()

function(add_mojo_package_doc target visibility package_dir output_directory)
  cmake_parse_arguments(KGEN "" "" "LINK_LIBS;LINK_DIRS;DEPENDS" ${ARGN})

  _setup_libs()

  get_filename_component(package_name ${package_dir} NAME_WE)
  set(GEN_DOC_JSON_FILE ${MODULAR_BINARY_DIR}/docs/${output_directory}${package_name}.mojodoc.json)
  set(GEN_DOC_DIR ${MODULAR_BINARY_DIR}/docs/${output_directory}${package_name})
  set(GEN_DOC_FILE ${GEN_DOC_DIR}/index.md)

  # Generate the JSON file containing the documentation description.
  add_custom_command(
    OUTPUT "${GEN_DOC_JSON_FILE}"

    COMMAND $<TARGET_FILE:mojo> doc
    -o ${GEN_DOC_JSON_FILE} ${KGEN_INCLUDE_FLAGS} ${package_dir}

    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    DEPENDS mojo-sdk ${KGEN_LIB_DEPENDS} ${KGEN_DEPENDS}
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
      -o ${GEN_DOC_DIR} ${GEN_DOC_JSON_FILE}

    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )

  add_custom_target(${target} DEPENDS ${GEN_DOC_FILE})
  set(is_public FALSE)
  if("${visibility}" STREQUAL "PUBLIC")
    set(is_public TRUE)
  endif()

  if (is_public)
    add_dependencies(modular-doc ${target})
  else()
    add_dependencies(modular-internal-doc ${target})
  endif()

  # Invoke `mojo --warn-missing-doc-strings` on the given file and pipe any
  # stderr warnings to a file.
  set(DOC_VALIDATE_WARNINGS_FILE ${GEN_DOC_JSON_FILE}.warnings.txt)
  add_custom_command(
    OUTPUT "${DOC_VALIDATE_WARNINGS_FILE}"

    COMMAND ${CMAKE_COMMAND} -E remove ${DOC_VALIDATE_WARNINGS_FILE}
    COMMAND $<TARGET_FILE:mojo> doc -warn-missing-doc-strings -o /dev/null
      ${KGEN_INCLUDE_FLAGS} ${package_dir} > "${DOC_VALIDATE_WARNINGS_FILE}" 2>&1

    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    DEPENDS mojo-sdk ${KGEN_LIB_DEPENDS} ${KGEN_DEPENDS}
  )

  # Add a `-validate` target that checks the `mojo doc -validate` output is
  # empty.
  set(failure_message
      "warnings were output when was run on '${package_name}'")
  add_custom_target("${target}-validate"
    COMMENT "Checking `mojo doc -warn-missing-doc-strings '${package_dir}'`..."
    COMMAND "${Python3_EXECUTABLE}"
      "${MODULAR_SOURCE_DIR}/KGEN/utils/check-file-is-empty.py"
      "${DOC_VALIDATE_WARNINGS_FILE}"
    DEPENDS "${DOC_VALIDATE_WARNINGS_FILE}"
  )

  # Only add PUBLIC Mojo files to the top-level doc validation target.
  # This ensures that `check-docs` only fails when our public
  # libraries are legitimately missing documentation.
  if (is_public)
    add_dependencies(check-docs "${target}-validate")
  endif()
endfunction()

function(add_internal_mojo_doc target doc_filename output_file output_directory)
  cmake_parse_arguments(KGEN "" "" "LINK_LIBS;LINK_DIRS" ${ARGN})

  _setup_libs()

  # Sanitize the target so it does not contain slashes.
  string(REPLACE "/" "_" target "${target}")

  set(GEN_DOC_JSON_FILE ${MODULAR_BINARY_DIR}/docs/${output_directory}${output_file}.mojodoc.json)
  set(GEN_DOC_FILE ${MODULAR_BINARY_DIR}/docs/${output_directory}${output_file}.md)

  # Generate the JSON file containing the documentation description.
  add_custom_command(
    OUTPUT "${GEN_DOC_JSON_FILE}"

    COMMAND $<TARGET_FILE:mojo> doc
    -o ${GEN_DOC_JSON_FILE} ${KGEN_INCLUDE_FLAGS} ${doc_filename}

    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    DEPENDS mojo-sdk ${doc_filename}
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

  add_custom_target(${target}DocGen DEPENDS ${GEN_DOC_FILE})
  add_dependencies(modular-internal-doc ${target}DocGen)
endfunction()

# Testing infrastructure
add_subdirectory(test)
add_subdirectory(unittests)
