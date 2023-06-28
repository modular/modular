##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

set(LLVM_LINK_COMPONENTS
  Support
  TableGen
)

add_tablegen(mojo-tblgen MOJO
  mojo-tblgen.cpp
  DriverCommand.cpp
  GenManPage.cpp)
