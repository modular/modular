##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_driver_option_tablegen(Mojo BuildOptions.td
  MANUAL_NAME mojo-build
  MARKDOWN_NAME build
  EXTRA_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/..")
