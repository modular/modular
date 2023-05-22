##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(MojoLLDB SHARED
  Plugin.cpp

  Commands/CommandObjectLLVMDebug.cpp
  Commands/CommandObjectMojo.cpp
  ExpressionParser/JITExecutionUnit.cpp
  ExpressionParser/JITUserExpression.cpp
  ExpressionParser/MojoExpressionParser.cpp
  ExpressionParser/MojoExpressionVariable.cpp
  ExpressionParser/MojoUserExpression.cpp
  REPL/MojoREPL.cpp
  TypeSystem/MojoTypeSystem.cpp

  DEPENDS
  mojo-python-interop

  LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsInfos
  ExecutionEngine
  MCJIT

  LINK_LIBS
  KGENCompiler
  KGENEmitObject
  LLCLRuntimeGlobals
  MSupportGlobals
  MVersion
  MojoParser

  MLIRBuiltinToLLVMIRTranslation

  liblldb
)

# Ensure symbols from the plugin library are visible.
set_target_properties(MojoLLDB PROPERTIES CXX_VISIBILITY_PRESET default)

# This creates an lldbinit file that is read automatically by our build of LLDB
# and is used to load the MojoLLDB plugin at startup.
set(lldbinit ${CMAKE_BINARY_DIR}/lib/lldbinit)
# The user can define `-DMOJO_REPL_DEBUG_LOG_FILE=/path/to/file`.
if (NOT DEFINED MOJO_REPL_DEBUG_LOG_FILE)
  set(MOJO_REPL_DEBUG_LOG_FILE ${MODULAR_DERIVED_DIR}/.mojorepl/debug.log)
endif()

# Get the filename as an absolute path.
get_filename_component(logfile ${MOJO_REPL_DEBUG_LOG_FILE} ABSOLUTE)
# And pull out its directory so we can make the directory if needed.
get_filename_component(logfile_dir ${logfile} DIRECTORY)

# Make the .derived/.repl directory and create the file so LLDB can use it.
file(MAKE_DIRECTORY ${logfile_dir})
file(TOUCH ${logfile})

# Generate the lldbinit file automatically. This loads the plugin and enables the log.
file(GENERATE
  OUTPUT ${lldbinit}
  CONTENT "plugin load $<TARGET_FILE:MojoLLDB>\nlog enable -f ${logfile} lldb expr\n"
  TARGET MojoLLDB)

# Find python.
find_package(Python3 COMPONENTS Interpreter REQUIRED)

# Generate a launcher for the mojo repl.
file(GENERATE
  OUTPUT ${CMAKE_BINARY_DIR}/bin/mojo-repl
  INPUT ${CMAKE_CURRENT_SOURCE_DIR}/mojo-repl.py.in
  TARGET Python3::Interpreter
  FILE_PERMISSIONS OWNER_READ OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

add_dependencies(MojoLLDB mojo-repl-entry-point)
