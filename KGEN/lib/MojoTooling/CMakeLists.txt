##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(MojoTooling
  ASTDeclRef.cpp
  ASTDeclView.cpp
  CodeComplete.cpp
  ParserDriver.cpp
  ParserDriverREPL.cpp
  REPLPythonExprUtils.cpp

  LINK_LIBS PUBLIC
  MojoParser
)
