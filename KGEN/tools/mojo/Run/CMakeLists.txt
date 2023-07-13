##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_driver_option_tablegen(Mojo RunOptions.td
  MANUAL_NAME mojo-run
  MARKDOWN_NAME run
  EXTRA_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/..")
