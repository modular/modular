##===- CMakeLists.txt -----------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_subdirectory(Benchmark)
add_subdirectory(ML)

# Get the commit hash of the current branch and write it as a macro to VCSRevision.h.
execute_process(
  COMMAND git log -1 --format=%H
  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
  OUTPUT_VARIABLE MODULAR_GIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/VCSRevision.h.in
  ${CMAKE_BINARY_DIR}/Support/include/Support/VCSRevision.h @ONLY)

add_modular_library(MSupport
  AlignedAlloc.cpp
  BlobCache.cpp
  ConcatenationTree.cpp
  CommonCLOptions.cpp
  Error.cpp
  HMAC.cpp

  ADDITIONAL_HEADER_DIRS

  DEPENDS

  LINK_COMPONENTS

  LINK_LIBS PUBLIC
  LLVMSupport
  )
