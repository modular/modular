##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_driver_option_tablegen(Mojo PackageOptions.td mojo-package
  EXTRA_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/..")
