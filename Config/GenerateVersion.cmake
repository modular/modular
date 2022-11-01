##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##
#
# This file contains the recipe for generating the Modular version file.
#
##===----------------------------------------------------------------------===##

find_package(Git QUIET REQUIRED)

execute_process (
    COMMAND ${GIT_EXECUTABLE} rev-parse --short=8 HEAD
    WORKING_DIRECTORY ${MODULAR_SOURCE_DIR}
    OUTPUT_VARIABLE MODULAR_VERSION_REVISION
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

set(version_inc "${MODULAR_BINARY_DIR}/Config/include/Config/Version.h.inc")

file(APPEND "${version_inc}.tmp"
  "/* Major version */\n"
  "#define MODULAR_VERSION_MAJOR ${MODULAR_VERSION_MAJOR}\n\n"

        "/* Minor version */\n"
  "#define MODULAR_VERSION_MINOR ${MODULAR_VERSION_MINOR}\n\n"

        "/* Patch version */\n"
  "#define MODULAR_VERSION_MAJOR ${MODULAR_VERSION_MAJOR}\n\n"

        "/* Revision sha */\n"
  "#define MODULAR_VERSION_REVISION \"${MODULAR_VERSION_REVISION}\"\n\n"

        "/* Version string */\n"
  "#define MODULAR_VERSION_STRING \"${MODULAR_VERSION_MAJOR}.${MODULAR_VERSION_MINOR}.${MODULAR_VERSION_PATCH}-${MODULAR_VERSION_REVISION}\"\n"
)

# Copy the file only if it has changed.
execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different
  "${version_inc}.tmp" "${version_inc}")
file(REMOVE "${version_inc}.tmp")
