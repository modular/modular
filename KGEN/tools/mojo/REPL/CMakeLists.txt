##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_driver_option_tablegen(Mojo REPLOptions.td mojo-repl
  EXTRA_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/..")

# The `repl` subcommand depends on Mojo's LLDB REPL.
if(LLDB_FOUND)
  add_dependencies(mojo MojoLLDB)
endif()
