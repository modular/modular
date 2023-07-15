##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_tool(mojo
  mojo.cpp
  Build/mojo-build.cpp
  Common/Compilation.cpp
  Common/Telemetry.cpp
  Demangle/mojo-demangle.cpp
  Doc/mojo-doc.cpp
  Package/mojo-package.cpp
  REPL/mojo-repl.cpp
  Run/mojo-run.cpp

  DEPENDS
  orc-rt-cas
  compiler-rt-cas
  KGENCompilerRT-static

  LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsInfos
  OrcTargetProcess

  LINK_LIBS PRIVATE
  LITDialect
  LLCLRuntimeGlobals
  MojoParser
  MSupportDriver
  MSupportGlobals
  KGENExecutionEngine
  KGENCompiler
  KGENEmitObject

  MLIRTransformUtils
  MLIRIR
  MLIRBuiltinToLLVMIRTranslation

  LLVMOption
)

# An umbrella target for all driver TableGen targets.
add_custom_target(MojoDriverOptions)

# Add targets to generate option definitions for the top-level driver.
add_driver_option_tablegen(Mojo DriverOptions.td
  MANUAL_NAME mojo
  MARKDOWN_NAME index
  EXTRA_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR})
# Each subcommand generates its own option definitions.
add_subdirectory(Build)
add_subdirectory(Demangle)
add_subdirectory(Doc)
add_subdirectory(Package)
add_subdirectory(REPL)
add_subdirectory(Run)

# Copy all generated Markdown files to a convenient location.
collect_driver_markdown(MojoDriverOptions)

# The driver depends on these option definitions.
target_include_directories(mojo PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
add_dependencies(mojo MojoDriverOptions)
