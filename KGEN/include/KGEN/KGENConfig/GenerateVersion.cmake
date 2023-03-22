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
set(KGEN_DEPEDENCIES
  ${MODULAR_SOURCE_DIR}/KGEN/include
  ${MODULAR_SOURCE_DIR}/KGEN/lib
  ${MODULAR_SOURCE_DIR}/KGEN/tools

  ${MODULAR_SOURCE_DIR}/Cache/include
  ${MODULAR_SOURCE_DIR}/Cache/lib

  ${MODULAR_SOURCE_DIR}/Support/include/Support/Compiler
  ${MODULAR_SOURCE_DIR}/Support/include/Support/DebugInfoDialect
  ${MODULAR_SOURCE_DIR}/Support/include/Support/HLCFDialect
  ${MODULAR_SOURCE_DIR}/Support/include/Support/HLCFToLLVM
  ${MODULAR_SOURCE_DIR}/Support/include/Support/Interpreter
  ${MODULAR_SOURCE_DIR}/Support/include/Support/MDialect

  ${MODULAR_SOURCE_DIR}/Support/lib/Compiler
  ${MODULAR_SOURCE_DIR}/Support/lib/DebugInfoDialect
  ${MODULAR_SOURCE_DIR}/Support/lib/HLCFDialect
  ${MODULAR_SOURCE_DIR}/Support/lib/HLCFToLLVM
  ${MODULAR_SOURCE_DIR}/Support/lib/Interpreter
  ${MODULAR_SOURCE_DIR}/Support/lib/MDialect

  ${MODULAR_SOURCE_DIR}/third-party/llvm-project
)

execute_process (
    COMMAND ${GIT_EXECUTABLE} rev-list -1 HEAD -- ${KGEN_DEPEDENCIES}
    WORKING_DIRECTORY ${MODULAR_SOURCE_DIR}
    OUTPUT_VARIABLE KGEN_VERSION_REVISION
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

set(version_inc "${MODULAR_BINARY_DIR}/KGEN/include/KGEN/KGENConfig/Version.h.inc")

configure_file(
  ${MODULAR_SOURCE_DIR}/KGEN/include/KGEN/KGENConfig/Version.h.in
  ${version_inc}
)
