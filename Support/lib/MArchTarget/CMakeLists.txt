##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(MArchTarget
  MArchTarget.cpp

  LINK_LIBS
  clangBasic
  MDialect
)

target_include_directories(MArchTarget PRIVATE ${CLANG_INCLUDE_DIRS})
