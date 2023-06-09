##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_tool(mojo-driver
  mojo-demangle.cpp
  mojo-driver.cpp

  LINK_LIBS PRIVATE
  LITDialect
  MSupportGlobals

  MLIRIR
)
