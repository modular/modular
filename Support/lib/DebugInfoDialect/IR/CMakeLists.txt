##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_dialect_library(DebugInfoDialect
  DebugInfoAttrs.cpp
  DebugInfoDialect.cpp
  DebugInfoOps.cpp
  DebugInfoTypes.cpp

  DEPENDS
  MLIRDebugInfoIncGen

  LINK_LIBS PUBLIC
  LLVMBinaryFormat
  MLIRIR
)
