##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_driver_option_tablegen(DocOptions.td)
add_driver_help_text_tablegen(DocOptions.td)
add_driver_man_page_tablegen(DocOptions.td mojo-driver-doc)
