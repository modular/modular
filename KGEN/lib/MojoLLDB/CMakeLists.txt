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
  Language/Formatters/MojoDecoratorBasedTypeFormatter.cpp
  Language/Formatters/MojoListTypeFormatter.cpp
  Language/Formatters/MojoKGENVariantTypeFormatter.cpp
  Language/MojoLanguage.cpp
  Language/MojoLanguageRuntime.cpp
  Logging/MojoExpressionLogger.cpp
  REPL/MojoREPL.cpp
  TypeSystem/MojoDWARFParser.cpp
  TypeSystem/MojoTypeDataLayout.cpp
  TypeSystem/MojoTypeSystem.cpp

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
  KGENObjectCompiler
  KGENSupport
  AsyncRTCompilerSupport
  AsyncRTRuntimeGlobals
  MCrashReporting
  MSupportGlobals
  MAsyncRTInit
  MVersion
  MojoTooling

  liblldb
  liblldbBreakpoint
  liblldbUtility
)

# This is needed to expose the headers of the LLDB plugins, which are based on
# the root of the lldb project.
target_include_directories(
  MojoLLDB PRIVATE
  "${MODULAR_SOURCE_DIR}/third-party/llvm-project/lldb/source/")

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
