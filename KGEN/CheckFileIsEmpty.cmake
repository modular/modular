##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

# A CMake script that raises a fatal error if the file at `PATH` is not empty.
# The error message is `FAILURE_MESSAGE`, followed by the contents of the file.

file(READ "${PATH}" contents)
string(STRIP "${contents}" contents)

if(NOT contents STREQUAL "")
  message(FATAL_ERROR "${FAILURE_MESSAGE}\n${contents}")
endif()
