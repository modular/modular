##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_driver_option_tablegen(Mojo BuildProjectOptions.td
  MANUAL_NAME mojo-build-project
  MARKDOWN_NAME build-project
  EXTRA_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/..")
