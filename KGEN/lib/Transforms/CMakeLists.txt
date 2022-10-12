##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_conversion_library(KGENTransforms
  KnownVariant.cpp

  LINK_LIBS PUBLIC
  KGENDialect
  POPDialect
)
