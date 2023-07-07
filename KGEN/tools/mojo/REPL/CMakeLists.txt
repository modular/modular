##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_driver_option_tablegen(Mojo REPLOptions.td mojo-repl)

# The `repl` subcommand depends on Mojo's LLDB REPL.
if(LLDB_FOUND)
  add_dependencies(mojo MojoLLDB)
endif()
