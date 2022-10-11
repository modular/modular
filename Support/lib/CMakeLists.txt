##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

find_package(Git REQUIRED)

add_subdirectory(Benchmark)
add_subdirectory(IndexDialect)
add_subdirectory(IndexToLLVM)
add_subdirectory(MDialect)
add_subdirectory(ML)
add_subdirectory(Compiler)

# Get the commit hash of the current branch and write it as a string literal in
# to VCSRevision.h.
execute_process(
  COMMAND ${GIT_EXECUTABLE} log -1 --format=%H
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
  TimeProfiler.cpp

  LINK_LIBS PUBLIC
  LLVMSupport
  )
