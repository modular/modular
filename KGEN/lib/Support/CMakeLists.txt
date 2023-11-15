##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(KGENSupport
  Configuration.cpp
  NameMangling.cpp

  LINK_LIBS PUBLIC
  MLIRIR
)
