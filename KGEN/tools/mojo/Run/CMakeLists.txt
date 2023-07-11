##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_driver_option_tablegen(Mojo RunOptions.td mojo-run
  EXTRA_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/..")
