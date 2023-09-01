##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_dialect_library(DebugInfoDialect
  DIBuilder.cpp
  DebugInfoAttrs.cpp
  DebugInfoDialect.cpp
  DebugInfoInterfaces.cpp
  DebugInfoOps.cpp
  DebugInfoTypes.cpp

  DEPENDS
  MLIRDebugInfoIncGen

  LINK_LIBS PUBLIC
  LLVMBinaryFormat
  MLIRIR
)
