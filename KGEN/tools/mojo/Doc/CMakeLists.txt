##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_driver_option_tablegen(Mojo DocOptions.td
  MANUAL_NAME mojo-doc
  MARKDOWN_NAME doc
  EXTRA_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/..")
