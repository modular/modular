##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(MojoLLDB SHARED
  Plugin.cpp

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

  LINK_LIBS
  KGENCompiler
  KGENEmitObject
  LLCLRuntimeGlobals
  ParseLit

  MLIRBuiltinToLLVMIRTranslation

  liblldb
)

# This creates an lldbinit file that is read automatically by our build of LLDB
# and is used to load the MojoLLDB plugin at startup.
set(lldbinit ${CMAKE_BINARY_DIR}/lib/lldbinit)

add_custom_command(
  OUTPUT ${lldbinit}
  COMMAND ${CMAKE_COMMAND} -E
    echo "plugin load $<TARGET_FILE:MojoLLDB>" > ${lldbinit}
)
add_custom_target(setup_lldbinit ALL DEPENDS ${lldbinit})

# The custom target MojoREPL collects all the dependencies needed for LLDB to
# automatically support Mojo.
add_custom_target(MojoREPL)
add_dependencies(MojoREPL setup_lldbinit MojoLLDB mojo-repl-entry-point)
