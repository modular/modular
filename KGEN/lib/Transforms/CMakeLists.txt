##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_conversion_library(KGENTransforms
  KnownVariant.cpp
  Mem2Reg.cpp

  LINK_LIBS PUBLIC
  KGENDialect
  POPDialect
)
