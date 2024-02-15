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

# MODULAR_REDACT_VERSION_REVISION can be set to avoid re-linking when switching
# between different commits, saving some time when committing frequently, at
# the expense of revision information being unavailable.  The safe default is
# OFF -- turn it on only if needed, and turn it back off before reporting any
# issues.
if(MODULAR_REDACT_VERSION_REVISION)
  set(MODULAR_VERSION_REVISION "<redacted>")
  message(STATUS
    "Redacting Modular version revision -- "
    "Do not distribute the resulting binaries, "
    "and turn off before reporting issues")
else()
  execute_process (
      COMMAND ${GIT_EXECUTABLE} rev-parse --short=8 HEAD
      WORKING_DIRECTORY ${MODULAR_SOURCE_DIR}
      OUTPUT_VARIABLE MODULAR_VERSION_REVISION
      OUTPUT_STRIP_TRAILING_WHITESPACE
  )
endif()

set(version_inc "${VERSION_OUTPUT_FILE}")

if(MODULAR_RELEASE_PACKAGE_BUILD)
  set(MODULAR_VERSION_STRING "\"${MODULAR_VERSION_MAJOR}.${MODULAR_VERSION_MINOR}.${MODULAR_VERSION_PATCH}${MODULAR_VERSION_LABEL}\"")
else()
  set(MODULAR_VERSION_STRING "\"${MODULAR_VERSION_MAJOR}.${MODULAR_VERSION_MINOR}.${MODULAR_VERSION_PATCH}${MODULAR_VERSION_LABEL}-${MODULAR_VERSION_REVISION}-${MODULAR_BUILD_TYPE_LOWER}\"")
endif()

file(APPEND "${version_inc}.tmp"
  "/* Major version */\n"
  "#define MODULAR_VERSION_MAJOR ${MODULAR_VERSION_MAJOR}\n\n"

  "/* Minor version */\n"
  "#define MODULAR_VERSION_MINOR ${MODULAR_VERSION_MINOR}\n\n"

  "/* Patch version */\n"
  "#define MODULAR_VERSION_PATCH ${MODULAR_VERSION_PATCH}\n\n"

  "/* Version for label */\n"
  "#define MODULAR_VERSION_LABEL \"${MODULAR_VERSION_LABEL}\"\n\n"

  "/* Revision sha */\n"
  "#define MODULAR_VERSION_REVISION \"${MODULAR_VERSION_REVISION}\"\n\n"

  "/* Build type */\n"
  "#define MODULAR_BUILD_TYPE_LOWER \"${MODULAR_BUILD_TYPE_LOWER}\"\n\n"

  "/* Version string */\n"
  "#define MODULAR_VERSION_STRING ${MODULAR_VERSION_STRING}\n"
  )

# Copy the file only if it has changed.
execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different
  "${version_inc}.tmp" "${version_inc}")
file(REMOVE "${version_inc}.tmp")
