##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

find_package(Python3 COMPONENTS Development)

add_modular_tool(mojo-repl-entry-point
  main.cpp

  LINK_LIBS PUBLIC
  KGENCompilerRT
  LLCLRuntimeGlobals
  MSupportGlobals
)

if (Python3_FOUND)
  target_link_libraries(mojo-repl-entry-point PUBLIC Python3::Python)
endif()

# This target is supposed to be located next to libMojoLLDB.
set_target_properties(mojo-repl-entry-point PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/"
)
