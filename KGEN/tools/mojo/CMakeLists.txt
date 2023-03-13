##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_tool(mojo
  mojo.cpp

  LINK_COMPONENTS
  native
  nativecodegen

  LINK_LIBS PRIVATE
  KGENCLOptions
  KGENCompiler
  KGENCompilerRT
  KGENDialect
  LITDialect
  POPDialect

  MLIRBytecodeWriter
  MLIRIndexDialect
  MLIRParser
  )

# TODO: This is a temporary target that is meant to keep both lit and mojo as
# valid executables.
add_custom_command(TARGET mojo
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:mojo> ${MODULAR_TOOLS_DIR}/lit)
