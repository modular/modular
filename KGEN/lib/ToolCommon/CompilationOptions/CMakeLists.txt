##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(KGENCompilationOptions
  CompilationOptions.cpp

  LINK_LIBS PUBLIC
  MDialect
  DebugInfoDialect
  KGENDialect
  )
