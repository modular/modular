##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_driver_option_tablegen(Mojo BuildOptions.td mojo-build
  EXTRA_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/..")
