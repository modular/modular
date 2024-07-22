##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(KGENSupport
  CompilerProfiling.cpp
  Configuration.cpp
  DebugInfoEncoding.cpp
  Debugging.cpp
  NameMangling.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MSupportSettings
)

target_compile_definitions(KGENSupport
  PRIVATE -DMOJO_CONFIG_SECTION=${MOJO_CONFIG_SECTION}
)

add_modular_library(KGENSupportForceLinkMLIRC
  ForceLinkMLIRC/ForceLinkMLIRC.cpp

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRCAPITransforms
)
