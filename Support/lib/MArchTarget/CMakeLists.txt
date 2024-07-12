##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

# This is the version used by modular internal targets.
add_modular_library(MArchTarget
  Host.cpp
  MArchTargetMinimal.cpp
  MArchTarget.cpp

  DEPENDS
  MLIRMIncGen

  LINK_LIBS
  clangBasic
  MDialect
  MThreading
  )

target_include_directories(MArchTarget PRIVATE ${CLANG_INCLUDE_DIRS})

# This is the version used by KGENCompilerRT-static for linking with external
# code, and its dependencies are intentionally kept minimal.
add_modular_library(MArchTargetMinimal
  OBJECT # Always generate an obj.${name} target.

  Host.cpp
  MArchTargetMinimal.cpp

  DEPENDS
  MLIRMIncGen

  LINK_LIBS
  clangBasic
  MThreading
)
target_include_directories(MArchTargetMinimal PRIVATE ${CLANG_INCLUDE_DIRS})
