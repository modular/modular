##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

# On MSVC compile with unicode sources so that we display 🔥 correctly.
if (MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /utf-8")
endif ()

add_modular_library(MojoLLDB SHARED
  Plugin.cpp

  Commands/CommandObjectLLVMDebug.cpp
  Commands/CommandObjectMojo.cpp
  ExpressionParser/JITExecutionUnit.cpp
  ExpressionParser/JITUserExpression.cpp
  ExpressionParser/MojoExpressionParser.cpp
  ExpressionParser/MojoExpressionVariable.cpp
  ExpressionParser/MojoUserExpression.cpp
  Language/MojoDecoratorBasedTypeFormatter.cpp
  Language/MojoDynamicVectorTypeFormatter.cpp
  Language/MojoLanguage.cpp
  Logging/MojoExpressionLogger.cpp
  REPL/MojoREPL.cpp
  TypeSystem/MojoTypeDataLayout.cpp
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
  KGENToolCommon
  KGENCompiler
  KGENEmitObject
  LLCLRuntimeGlobals
  MCrashReporting
  MSupportGlobals
  MVersion
  MojoTooling

  liblldb
)

# Ensure symbols from the plugin library are visible.
set_target_properties(MojoLLDB PROPERTIES CXX_VISIBILITY_PRESET default)
target_compile_definitions(MojoLLDB PRIVATE MODULAR_BUILDING_LIBRARY)

# We used to create an lldbinit file to automatically load our MojoLLDB plugin,
# but this conflicted with the --one-line-before-file load of the plugin
# provided by the 'mojo repl' command.  We do not want to generate this file
# any more.  Furthermore, we also want to remove stale copies of the file from
# Modulers' build directories, because otherwise it will stick around and
# continue to cause issues.
set(lldbinit ${CMAKE_BINARY_DIR}/lib/lldbinit)
file(REMOVE ${lldbinit})

add_dependencies(MojoLLDB mojo-repl-entry-point)
