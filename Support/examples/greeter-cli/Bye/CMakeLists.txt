##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##
#
# The contents of this file are explained in `../CMakeLists.txt`.
#
##===----------------------------------------------------------------------===##

add_driver_option_tablegen(GreeterCLI ByeOptions.td
    MANUAL_NAME greeter-cli-bye
    MARKDOWN_NAME bye
    EXTRA_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/..)
