##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_driver_option_tablegen(Mojo DemangleOptions.td mojo-demangle
  EXTRA_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/..")
