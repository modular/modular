##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

include(AddPython)
modular_find_python3_development()

# `mojo repl` calls out to `lldb` in a subprocess (for now). As such, it needs
# to locate the `lldb` executable to use, and currently this is done by invoking
# an `lldb` that resides in the same directory as `mojo`. `lldb` itself locates
# the dynamic libraries it relies upon by searching paths relative to itself.
# LLVM's build places these libraries at the correct paths, so, rather than
# copying `lldb` to a location alongside `mojo` (and thus breaking the relative
# paths), create a symlink to its location in the LLVM build tree.
find_program(lldb_path "lldb" PATHS "${LLVM_BUILD_DIR}/bin" NO_DEFAULT_PATH)
add_custom_target(mojo-lldb
  COMMAND
    ${CMAKE_COMMAND} -E create_symlink ${lldb_path} ${MODULAR_TOOLS_DIR}/lldb)

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
  mojo-lldb
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
  LLCLRuntime
  MArchTarget
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
if(WIN32)
  set(mblackPathSeparator "\;")
endif()

if(MODULAR_PYTHON_EXECUTABLE STREQUAL "")
  set(PYTHON_EXE ${Python3_EXECUTABLE})
else()
  set(PYTHON_EXE ${MODULAR_PYTHON_EXECUTABLE})
endif()

message(STATUS "Using ${PYTHON_EXE} to create mblack binary.")

if (MODULAR_RELEASE_PACKAGE_BUILD)
  message(STATUS "Skipping binary mblack build for release package build.")
  return()
endif()

add_custom_target(mblack
  ${CMAKE_COMMAND} -E env PYI_LOG_LEVEL=ERROR
  ${PYTHON_EXE} -m PyInstaller -F --name mblack
  --add-data "${MODULAR_UTILS_DIR}/mblack/src/mblib2to3${mblackPathSeparator}mblib2to3"
  "${MODULAR_UTILS_DIR}/mblack/src/mblack/__main__.py"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  ${CMAKE_CURRENT_BINARY_DIR}/dist/mblack${CMAKE_EXECUTABLE_SUFFIX}
  ${CMAKE_BINARY_DIR}/bin/
  )
add_dependencies(mojo mblack)
