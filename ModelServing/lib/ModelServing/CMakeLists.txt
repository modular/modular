##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(MCUCalculator
  UnitCalculator.cpp

  LINK_LIBS PUBLIC
  MSupport
)

if(MODULAR_WITH_AWS_BILLING)
  add_subdirectory(AWS)
endif()
