##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

# This is a workaround/ fix to make python wheels packing work
# without the following line cmake can't find python development in
# modular_find_python3_development() macro.
# Potentially it's a cmake issue or/and how we use it
# (e.g. call find_package(Python3 ...) multiple times)
find_package(Python3 COMPONENTS Development)

include(AddPython)
modular_find_python3_development()

set(sources
  mojo.cpp
  Build/mojo-build.cpp
  Common/Compilation.cpp
  Common/LLDB.cpp
  Demangle/mojo-demangle.cpp
  Doc/mojo-doc.cpp
  Format/mojo-format.cpp
  Debug/mojo-debug.cpp
  Debug/RPCServer.cpp
  Package/mojo-package.cpp
  REPL/mojo-repl.cpp
  Run/mojo-run.cpp
  Test/mojo-test.cpp)

set(tablegen_definitions)
if (MODULAR_MOJO_BUILD_SERVER_PREVIEW)
  set(sources ${sources} BuildProject/mojo-build-project.cpp)
  list(APPEND tablegen_definitions MODULAR_MOJO_BUILD_SERVER_PREVIEW)
endif()

add_modular_tool(mojo
  ${sources}

  DEPENDS
  KGENCompilerRT-static
  modular-crashpad-handler
  MojoLLDB

  LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsInfos
  OrcTargetProcess

  LINK_LIBS PRIVATE
  KGENToolCommon
  LLCLRuntimeGlobals
  MArchTarget
  MCrashReporting
  MojoToolsCommon
  MojoParser
  MojoBuild
  MojoTesting
  MojoTooling
  MSupportFilesystem
  MSupportDriver
  MSupportGlobals
  MSupportInit
  KGENExecutionEngine
  KGENCompiler
  KGENEmitObject

  MLIRTransformUtils
  MLIRIR

  LLVMOption
  )

# An umbrella target for all driver TableGen targets.
add_custom_target(MojoDriverOptions)

# Add targets to generate option definitions for the top-level driver.
add_driver_option_tablegen(Mojo DriverOptions.td
  MANUAL_NAME mojo
  MARKDOWN_NAME index
  EXTRA_DEFINITIONS ${tablegen_definitions}
  EXTRA_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR})
# Each subcommand generates its own option definitions.
add_subdirectory(Build)
add_subdirectory(Debug)
add_subdirectory(Demangle)
add_subdirectory(Doc)
add_subdirectory(Format)
add_subdirectory(Package)
add_subdirectory(REPL)
add_subdirectory(Run)
add_subdirectory(Test)
if(MODULAR_MOJO_BUILD_SERVER_PREVIEW)
  add_subdirectory(BuildProject)
endif()

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

if (MODULAR_STATIC_LINKED_PYTHON)
  message(STATUS "Skipping binary mblack build when Python is statically linked (e.g. for building wheels inside manylinux).")
  return()
endif()

set(MBLACK_BIN "${CMAKE_BINARY_DIR}/lib/mblack/mblack${CMAKE_EXECUTABLE_SUFFIX}")
set(MBLACK_SRC_DIR "${MODULAR_UTILS_DIR}/mblack/src")
set(MBLACK_SRC "${MBLACK_SRC_DIR}/mblack/__main__.py")

add_custom_command(
  OUTPUT "${MBLACK_BIN}"
  COMMAND "${CMAKE_COMMAND}" -E env PYI_LOG_LEVEL=ERROR
    "${PYTHON_EXE}" -m PyInstaller -y -D --name mblack
    --add-data "${MBLACK_SRC_DIR}/mblib2to3${mblackPathSeparator}mblib2to3"
    --distpath "${CMAKE_BINARY_DIR}/lib"
    "${MBLACK_SRC}"
  DEPENDS "${MBLACK_SRC}" "${MBLACK_SRC_DIR}/mblib2to3/Grammar.txt"
  COMMENT "Building mblack."
  VERBATIM
  )

add_custom_target(mblack DEPENDS "${MBLACK_BIN}")

add_dependencies(mojo mblack mojo-test-executor)
