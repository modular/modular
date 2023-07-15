##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

add_modular_tool(mojo
  mojo.cpp
  Build/mojo-build.cpp
  Common/Compilation.cpp
  Common/Telemetry.cpp
  Demangle/mojo-demangle.cpp
  Doc/mojo-doc.cpp
  Format/mojo-format.cpp
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
add_subdirectory(Format)
add_subdirectory(Package)
add_subdirectory(REPL)
add_subdirectory(Run)

# Copy all generated Markdown files to a convenient location.
collect_driver_markdown(MojoDriverOptions)

# The driver depends on these option definitions.
target_include_directories(mojo PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
add_dependencies(mojo MojoDriverOptions)

# Generate a packaged version of mblack for use in the Mojo driver.
set(mblackPathSeparator ":")
if (WIN32)
  set(mblackPathSeparator ";")
endif()
add_custom_target(mblack
  ${CMAKE_COMMAND} -E env PYI_LOG_LEVEL=ERROR
    ${Python3_EXECUTABLE} -m PyInstaller -F --name mblack
      --add-data '${MODULAR_UTILS_DIR}/mblack/src/mblib2to3${mblackPathSeparator}mblib2to3'
      ${MODULAR_UTILS_DIR}/mblack/src/mblack/__main__.py
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_BINARY_DIR}/dist/mblack${CMAKE_EXECUTABLE_SUFFIX}
    ${CMAKE_BINARY_DIR}/bin/
)
add_dependencies(mojo mblack)
