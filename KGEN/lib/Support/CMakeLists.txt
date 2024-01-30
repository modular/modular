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

target_compile_definitions(KGENSupport
  PRIVATE -DMOJO_CONFIG_SECTION="${MOJO_CONFIG_SECTION}"
)
