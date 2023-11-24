##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

# Pass CMake values for dynamic library file paths into a header that can be
# used by the `repl` and `debug` subcommands.
configure_file(MojoDebug.h.in MojoDebug.h)

add_driver_option_tablegen(Mojo DebugOptions.td
  MANUAL_NAME mojo-debug
  MARKDOWN_NAME debug
  EXTRA_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/..")
