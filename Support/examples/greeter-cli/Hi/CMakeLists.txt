##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##
#
# The contents of this file are explained in `../CMakeLists.txt`.
#
##===----------------------------------------------------------------------===##

add_driver_option_tablegen(GreeterCLI HiOptions.td
    MANUAL_NAME greeter-cli-hi
    MARKDOWN_NAME hi
    EXTRA_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/..)
