##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(MSupportDriver
  DiagnosticFormat.cpp
  DriverSupport.cpp

  LINK_LIBS PRIVATE
  MSupportFilesystem
  MSupport
  LLVMOption
)
