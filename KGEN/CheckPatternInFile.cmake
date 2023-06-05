##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

# A CMake script that raises a fatal error if the regular expression `PATTERN`
# occurs within the contents of the file at `PATH`. The error message is
# `FAILURE_MESSAGE`, followed by each match of the regular expression.

file(READ "${PATH}" contents)
string(REGEX MATCHALL "${PATTERN}" results "${contents}")

message(VERBOSE "Searching for pattern \"${PATTERN}\" in \"${PATH}\"...")

if(results)
  foreach(result ${results})
    string(CONCAT FAILURE_MESSAGE "${FAILURE_MESSAGE}" "\n${result}\n")
  endforeach()

  message(FATAL_ERROR "${FAILURE_MESSAGE}")
endif()
