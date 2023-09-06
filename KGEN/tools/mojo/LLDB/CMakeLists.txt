##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

# Pass CMake values for dynamic library file paths into a header that can be
# used by the `repl` and `lldb` subcommands.
configure_file(MojoLLDB.h.in MojoLLDB.h)

add_driver_option_tablegen(Mojo LLDBOptions.td
  MANUAL_NAME mojo-lldb
  MARKDOWN_NAME lldb
  EXTRA_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/..")

# The `lldb` subcommand depends on Mojo's LLDB plugin.
if(LLDB_FOUND)
  add_dependencies(mojo MojoLLDB)
endif()
