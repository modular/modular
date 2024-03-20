##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_driver_option_tablegen(Mojo TestOptions.td
  MANUAL_NAME mojo-test
  MARKDOWN_NAME test
  EXTRA_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/..")
