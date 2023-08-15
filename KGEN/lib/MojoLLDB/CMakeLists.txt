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
  MCrashReporting
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

# Generate the lldbinit file automatically. This loads the plugin and enables the log.
file(GENERATE
  OUTPUT ${lldbinit}
  CONTENT "plugin load $<TARGET_FILE:MojoLLDB>\n"
  TARGET MojoLLDB)

add_dependencies(MojoLLDB mojo-repl-entry-point)
