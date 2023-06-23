##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_tool(mojo-driver
  mojo-driver.cpp
  Demangle/mojo-demangle.cpp
  Doc/mojo-doc.cpp

  DEPENDS
  orc-rt-cas
  compiler-rt-cas

  LINK_LIBS PRIVATE
  LITDialect
  LLCLRuntimeGlobals
  MojoParser
  MSupportGlobals

  MLIRTransformUtils
  MLIRIR

  LLVMOption
)

# An umbrella target for all driver option defnition TableGen targets.
add_custom_target(MojoDriverOptions)

# Given an input path "Foo.td", this creates a public TableGen target named
# "MojoDriverFoo", which when built outputs option parser definitions to a file
# "Foo.inc". It also adds that target to the driver option umbrella target.
macro(add_driver_option_tablegen input)
  get_filename_component(name "${input}" NAME_WE)
  set(LLVM_TARGET_DEFINITIONS "${input}")
  tablegen(LLVM "${name}.inc" -gen-opt-parser-defs)
  add_public_tablegen_target("MojoDriver${name}")
  add_dependencies(MojoDriverOptions "MojoDriver${name}")
endmacro()

# Add targets to generate option definitions for the top-level driver.
add_driver_option_tablegen(DriverOptions.td)
# Each subcommand generates its own option definitions.
add_subdirectory(Demangle)
add_subdirectory(Doc)

# The driver depends on these option definitions.
target_include_directories(mojo-driver PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
add_dependencies(mojo-driver MojoDriverOptions)
