##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_driver_option_tablegen(REPLOptions.td)
add_driver_help_text_tablegen(REPLOptions.td)
add_driver_man_page_tablegen(REPLOptions.td mojo-driver-repl)

# The `repl` subcommand depends on Mojo's LLDB REPL.
if(LLDB_FOUND)
  add_dependencies(mojo-driver MojoLLDB)
endif()
