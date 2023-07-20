##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

# Pass CMake values for dynamic library file paths into a header that can be
# used by the `repl` subcommand.
configure_file(MojoLLDB.h.in MojoLLDB.h)

add_driver_option_tablegen(Mojo REPLOptions.td
  MANUAL_NAME mojo-repl
  MARKDOWN_NAME repl
  EXTRA_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/..")

# The `repl` subcommand depends on Mojo's LLDB REPL.
if(LLDB_FOUND)
  add_dependencies(mojo MojoLLDB)
endif()
