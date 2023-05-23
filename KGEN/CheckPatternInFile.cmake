##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

# A CMake script that raises a fatal error if the regular expression `PATTERN`
# occurs within the contents of the file at `PATH`.

file(READ "${PATH}" contents)
string(REGEX MATCH "${PATTERN}" result "${contents}")

if(NOT ${result} EQUAL -1)
  message(FATAL_ERROR "Pattern \"${PATTERN}\" appears in \"${PATH}\"")
endif()
