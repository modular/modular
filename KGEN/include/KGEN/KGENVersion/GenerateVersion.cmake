##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##
#
# This file contains the recipe for generating the KGEN version file.
#
##===----------------------------------------------------------------------===##

find_package(Git QUIET REQUIRED)

# Compute the set of folders that KGEN depends on.
set(KGEN_DEPENDENCIES
  ${MODULAR_SOURCE_DIR}/KGEN/include
  ${MODULAR_SOURCE_DIR}/KGEN/lib
  ${MODULAR_SOURCE_DIR}/KGEN/tools

  ${MODULAR_SOURCE_DIR}/Cache/include
  ${MODULAR_SOURCE_DIR}/Cache/lib

  ${MODULAR_SOURCE_DIR}/Support/include/Support/Compiler
  ${MODULAR_SOURCE_DIR}/Support/include/Support/DebugInfoDialect
  ${MODULAR_SOURCE_DIR}/Support/include/KGEN/HLCFDialect
  ${MODULAR_SOURCE_DIR}/Support/include/Support/Interpreter
  ${MODULAR_SOURCE_DIR}/Support/include/Support/MDialect

  ${MODULAR_SOURCE_DIR}/Support/lib/Compiler
  ${MODULAR_SOURCE_DIR}/Support/lib/DebugInfoDialect
  ${MODULAR_SOURCE_DIR}/Support/lib/Interpreter
  ${MODULAR_SOURCE_DIR}/Support/lib/MDialect

  # Changes to the builtin module will change the mojo ir being generated.
  ${MODULAR_SOURCE_DIR}/Kernels/mojo/stdlib/builtin

  ${MODULAR_SOURCE_DIR}/third-party/llvm-project
)

execute_process (
    COMMAND ${GIT_EXECUTABLE} ls-files --stage -- ${KGEN_DEPENDENCIES}
    WORKING_DIRECTORY ${MODULAR_SOURCE_DIR}
    OUTPUT_VARIABLE KGEN_FILE_HASHES
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

string(SHA256 KGEN_VERSION_REVISION ${KGEN_FILE_HASHES})

set(version_inc "${MODULAR_BINARY_DIR}/KGEN/include/KGEN/KGENVersion/Version.h.inc")

configure_file(
  ${MODULAR_SOURCE_DIR}/KGEN/include/KGEN/KGENVersion/Version.h.in
  ${version_inc}
)

set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${version_inc})
