##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(MSupportSettings
  Settings.cpp

  LINK_LIBS PUBLIC
  MSupport
  MSupportEntitlements
)

if (MODULAR_PREVIEW)
  add_compile_definitions(
    __SETTINGS_ONLY_MODULAR_PREVIEW=1
  )
endif()
