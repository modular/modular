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
add_subdirectory(lib/TransformUtils)
add_subdirectory(lib/CODialect)
add_subdirectory(lib/POPDialect)
add_subdirectory(lib/LITDialect)
add_subdirectory(lib/CustomDialect)

# KGEN MLIR passes.
add_subdirectory(lib/LowerLIT)
add_subdirectory(lib/KGENToLLVM)
add_subdirectory(lib/Transforms)
add_subdirectory(lib/Elaborator)

# KGEN->KGEN MOGG specific transforms.
add_subdirectory(lib/MOGGPreElab)

# KGEN compiler and JIT.
add_subdirectory(lib/Compiler/Pipeline)
add_subdirectory(lib/ToolCommon)
add_subdirectory(lib/CompilerRT)
add_subdirectory(lib/ExecutionEngine/JIT)
add_subdirectory(lib/ExecutionEngine)
add_subdirectory(lib/Compiler/ObjectCompiler)
add_subdirectory(lib/Compiler)

# Mojo frontend.
add_subdirectory(lib/MojoParser)
add_subdirectory(lib/MojoTooling)

add_subdirectory(lib/MojoBuild)
add_subdirectory(lib/MojoLLDB)
add_subdirectory(lib/MojoJupyter)
add_subdirectory(lib/MojoTesting)

add_subdirectory(tools/common)
add_subdirectory(tools/hash-mlir)
add_subdirectory(tools/kgen)
add_subdirectory(tools/kgen-opt)
add_subdirectory(tools/kgen-reduce)
add_subdirectory(tools/kgen-translate)
add_subdirectory(tools/llvm-module-split)
add_subdirectory(tools/mojo-lsp-server)
add_subdirectory(tools/mojo-lsp-test-client)

add_subdirectory(tools/mojo-jupyter-executor)
add_subdirectory(tools/mojo-repl-entry-point)
add_subdirectory(tools/mojo-test-executor)

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


# Add the necessary tool dependencies to the mojo-sdk target.
add_dependencies(mojo-sdk mojo mojo-lsp-server MojoJupyter)

# This function will generate the CMake targets needed to build a Mojo binary
# for the current source file.
#   Usage:
#
#  add_mojo_binary(mojo-binary
#     something_with_main.mojo
#  )
#
# This will generate a target and binary called `mojo-binary`.
#
# This function will pull its include flags from the linked libs to avoid
# having too many search paths.
function(add_mojo_binary name)
  cmake_parse_arguments(KGEN
                        ""
                        "OUTPUT_DIR;OUTPUT_NAME"
                        "LINK_LIBS;LINK_DIRS;DEPENDS" ${ARGN})

  _setup_libs()

  if (NOT KGEN_OUTPUT_DIR)
    set(KGEN_OUTPUT_DIR ${CMAKE_BINARY_DIR}/bin)
  else()
    # Make sure that the directory is created.
    file(MAKE_DIRECTORY ${KGEN_OUTPUT_DIR})
  endif ()

  set(KGEN_SOURCE_FILES "")
  foreach(file ${KGEN_UNPARSED_ARGUMENTS})
    list(APPEND KGEN_SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/${file}")
  endforeach()

  # Ensure at most one source file is specified
  if (NOT KGEN_SOURCE_FILES)
    message(FATAL_ERROR "Must specify one mojo source files when calling `add_mojo_binary`.")
  endif()
  list(LENGTH KGEN_SOURCE_FILES KGEN_SOURCE_FILES_COUNT)
  if (KGEN_SOURCE_FILES_COUNT GREATER 1)
    message(FATAL_ERROR "Must specify a single mojo source files when calling `add_mojo_binary`.")
  endif()

  # Add the command used to generate the archive for the kernel.
  add_custom_command(
    OUTPUT ${KGEN_OUTPUT_DIR}/${name}

    COMMAND $<TARGET_FILE:mojo> build
    ${KGEN_SOURCE_FILES} -o ${KGEN_OUTPUT_DIR}/${name} ${KGEN_INCLUDE_FLAGS}

    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    DEPENDS mojo-stdlib ${KGEN_SOURCE_FILES} ${KGEN_LIB_DEPENDS} ${KGEN_DEPENDS}
  )
  add_custom_target(${name} DEPENDS ${KGEN_OUTPUT_DIR}/${name})
endfunction()

# Given a target name, this function will collect the set of exposed Mojo
# packages that are dependencies of the target.
macro(collect_exposed_mojo_packages target names)
  foreach (name ${names})
    if (TARGET ${name})
      get_target_property(new_packages ${name} EXPOSED_MOJO_PACKAGES)
      if (new_packages)
        list(APPEND MOJO_DEPENDENCIES ${new_packages})
      endif()

      # Recurse into the dependencies.
      get_target_property(depends ${name} MOJO_DEPENDENCIES)
      if (depends)
        list(APPEND MOJO_DEPENDENCIES ${depends})
      endif()
    else()
      message(FATAL_ERROR
        "Unknown Mojo dependency ${name} of ${target}! Is the "
        "target defined before the dependency?"
      )
    endif()
  endforeach()
endmacro()

# Validate that a given package only depends on the set of packages that it
# declares as dependencies.
function(validate_mojo_dependencies target package_dir depends)
  # Run a python script to validate that all of the dependencies are within the
  # known set.
  set(DEPENDS_FILE ${CMAKE_CURRENT_BINARY_DIR}/${target}-validate-deps.txt)
  add_custom_command(
    OUTPUT ${DEPENDS_FILE}
    COMMENT "Validating Mojo dependencies for ${target}..."

    COMMAND ${Python3_EXECUTABLE}
      ${MODULAR_SOURCE_DIR}/KGEN/utils/validate_mojo_dependencies.py
      --path ${package_dir} --known_dependencies ${depends}
    COMMAND ${CMAKE_COMMAND} -E touch ${DEPENDS_FILE}

    DEPENDS
      ${MODULAR_SOURCE_DIR}/KGEN/utils/validate_mojo_dependencies.py
      ${package_dir}
  )
  add_custom_target(${target}-validate-deps DEPENDS ${DEPENDS_FILE})
  add_dependencies(${target} ${target}-validate-deps)
endfunction()

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
# documentation. Location of built mojopkg can be queried with the property
# `PACKAGE_LOCATION`.
#
# This function will pull its include flags from the linked libs to avoid
# having too many search paths.
function(add_mojo_package target visibility)
  cmake_parse_arguments(KGEN
                        "STDLIB"
                        "PACKAGE_DIR;PACKAGE_OUTPUT_DIR;OUTPUT_NAME;DOCS_DIR"
                        "LINK_LIBS;LINK_DIRS;DEPENDS" ${ARGN})

  _setup_libs()

  if (NOT KGEN_PACKAGE_DIR)
    set(KGEN_PACKAGE_DIR ${CMAKE_CURRENT_LIST_DIR})
  endif ()
  if (NOT KGEN_PACKAGE_OUTPUT_DIR)
    set(KGEN_PACKAGE_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
  else()
    # Make sure that the directory is created.
    file(MAKE_DIRECTORY ${KGEN_PACKAGE_OUTPUT_DIR})
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
  file(RELATIVE_PATH package_file_name_rel ${CMAKE_BINARY_DIR} ${KGEN_PACKAGE_OUTPUT_DIR}/${package_file_name})

  set(KGEN_SOURCE_FILES "")
  foreach(file ${KGEN_UNPARSED_ARGUMENTS})
    list(APPEND KGEN_SOURCE_FILES "${KGEN_PACKAGE_DIR}/${file}.mojo")
  endforeach()

  # Ensure at least one source file is specified
  if (NOT KGEN_SOURCE_FILES)
    message(FATAL_ERROR "Must specify one or more mojo source files when calling `add_mojo_package`.")
  endif()

  # If this is not a standard library package, then the package should depend
  # on the full-stdlib.
  if (NOT KGEN_STDLIB)
    list(APPEND KGEN_DEPENDS mojo-stdlib)
  endif()

  set(KGEN_PACKAGE_FILENAME "${KGEN_PACKAGE_DIR}/__init__.mojo")

  # Compute the set of exposed packages that are generated by this package. For
  # the `stdlib` this is the modules nested within, for everything else it's
  # just the output package.
  set(exposed_packages ${KGEN_OUTPUT_NAME})
  if (KGEN_STDLIB AND (${KGEN_OUTPUT_NAME} STREQUAL "stdlib"))
    foreach(module ${KGEN_UNPARSED_ARGUMENTS})
      string(REGEX MATCH "([^/]*)/.*" _ ${module})
      if (IS_DIRECTORY ${KGEN_PACKAGE_DIR}/${CMAKE_MATCH_1})
        list(APPEND exposed_packages ${CMAKE_MATCH_1})
      endif()
    endforeach()
    list(REMOVE_DUPLICATES exposed_packages)
  endif()

  # Add the command used to generate the package.
  add_custom_command(
    OUTPUT ${KGEN_PACKAGE_OUTPUT_DIR}/${package_file_name}

    COMMAND $<TARGET_FILE:mojo> package
    ${KGEN_PACKAGE_DIR} -o ${package_file_name_rel} ${KGEN_INCLUDE_FLAGS}

    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    DEPENDS mojo ${KGEN_SOURCE_FILES} ${KGEN_LIB_DEPENDS} ${KGEN_PACKAGE_FILENAME} ${KGEN_DEPENDS}
  )
  add_custom_target(${target} DEPENDS ${KGEN_PACKAGE_OUTPUT_DIR}/${package_file_name})

  # Compute the full set of dependencies for the package.
  set(MOJO_DEPENDENCIES "")
  collect_exposed_mojo_packages(${target} "${KGEN_DEPENDS}")
  list(REMOVE_DUPLICATES MOJO_DEPENDENCIES)

  set_target_properties(${target}
    PROPERTIES
    INCLUDE_DIRECTORIES ${KGEN_PACKAGE_OUTPUT_DIR}
    PACKAGE_LOCATION ${KGEN_PACKAGE_OUTPUT_DIR}/${package_file_name}
    EXPOSED_MOJO_PACKAGES "${exposed_packages}"
    MOJO_DEPENDENCIES "${MOJO_DEPENDENCIES}"
  )

  # If this is a standard library package, then we need to add it to the
  # mojo-sdk and mojo-stdlib target.
  if (KGEN_STDLIB)
    add_dependencies(mojo-stdlib ${target})
    install(FILES "${CMAKE_BINARY_DIR}/${package_file_name_rel}"
            DESTINATION "lib/mojo"
            COMPONENT mojo-sdk)

    # If this is a standard library package, add in the set of exposed packages
    # to the mojo-stdlib target.
    get_target_property(existing_exposed_packages mojo-stdlib EXPOSED_MOJO_PACKAGES)
    list(APPEND existing_exposed_packages ${exposed_packages})
    set_target_properties(mojo-stdlib PROPERTIES EXPOSED_MOJO_PACKAGES "${existing_exposed_packages}")

  # Generate documentation for the package. Standard library documentation is
  # generated elsewhere.
  else()
    add_mojo_package_doc(${target}-doc ${visibility} ${KGEN_PACKAGE_DIR}
      ${KGEN_DOCS_DIR}/ DEPENDS ${KGEN_SOURCE_FILES} ${target})
  endif()

  validate_mojo_dependencies(${target} ${KGEN_PACKAGE_DIR} "${MOJO_DEPENDENCIES}")
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
    "SAVE_TEMPS;NO_DEBUG"
    "OUTPUT"
    "LINK_LIBS;LINK_DIRS"
    ${ARGN})

  _setup_libs()
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
                        "NO_DEBUG"
                        "SOURCE;OUTPUT_NAME"
                        "KERNELS;DEPENDS;LINK_LIBS;LINK_DIRS" ${ARGN})

  _setup_libs()
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
    -o ${header_file_name_rel} ${KGEN_INCLUDE_FLAGS} ${KGEN_SOURCE}
    -d ${header_file_name_rel}.d
    -O${KGEN_OPTIMIZATION_LEVEL}
    -debug-level=${KGEN_DEBUG_LEVEL}
    ${KGEN_SANITIZER_OPTIONS}
    $<$<BOOL:${MODULAR_ENABLE_XRAY}>:-xray-instrument>

    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    DEPFILE ${CMAKE_CURRENT_BINARY_DIR}/${header_file_name}.d
    DEPENDS kgen ${KGEN_SOURCE} ${KGEN_LIB_DEPENDS} mojo-sdk ${KGEN_DEPENDS}
  )
  add_custom_target(${target}-headers DEPENDS ${header_file_name})
  add_dependencies(${target}-headers ${KGEN_LIB_DEPENDS} kgen mojo-sdk)
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
  set_target_properties(${target} PROPERTIES OUTPUT_DIR ${GEN_DOC_DIR})

  set(is_public FALSE)
  if("${visibility}" STREQUAL "PUBLIC")
    set(is_public TRUE)
  endif()

  if (is_public)
    add_dependencies(modular-doc ${target})
  else()
    add_dependencies(modular-internal-doc ${target})
  endif()

  # Add a `-validate` target that invokes `mojo doc --validate-doc-strings` on
  # the given package source directory.
  add_custom_target("${target}-validate"
    COMMENT "Checking `mojo doc --validate-doc-strings '${package_dir}'`..."
    COMMAND $<TARGET_FILE:mojo> doc --validate-doc-strings
      --diagnose-missing-doc-strings
      ${KGEN_INCLUDE_FLAGS} ${package_dir} -o /dev/null
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    DEPENDS mojo-sdk ${KGEN_LIB_DEPENDS} ${KGEN_DEPENDS}
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

add_subdirectory(tools/mojo-build-server)

# Testing infrastructure
add_subdirectory(test)
add_subdirectory(unittests)
