##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_driver_option_tablegen(Mojo FormatOptions.td
  MANUAL_NAME mojo-format
  MARKDOWN_NAME format
  EXTRA_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/..")
