##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_driver_option_tablegen(DemangleOptions.td)
add_driver_help_text_tablegen(DemangleOptions.td)
add_driver_man_page_tablegen(DemangleOptions.td mojo-driver-demangle)
