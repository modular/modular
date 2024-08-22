##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_driver_option_tablegen(Mojo DebugOptions.td
  MANUAL_NAME mojo-debug
  MARKDOWN_NAME debug
  EXTRA_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/..")
