##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

set(SOURCES
  InitIntelAMX.cpp
  Initialize.cpp
  LLCL.cpp
  Memory.cpp
  OutputChain.cpp
  Python.cpp
  Random.cpp
  System.cpp
  Tracing.cpp
)

# A statically built CompilerRT library.
add_modular_library(KGENCompilerRT
  ${SOURCES}

  LINK_LIBS PUBLIC
  MSupport
  LLCLRuntime
)

target_compile_definitions(KGENCompilerRT PRIVATE MODULAR_BUILDING_COMPILERRT)

# A dynamically built CompilerRT library used when jitting.
add_modular_library(KGENCompilerRTShared SHARED
  ${SOURCES}

  LINK_LIBS PRIVATE
  MSupport
  MSupportGlobals
  LLCLRuntime
  LLCLRuntimeGlobals
)

target_compile_definitions(KGENCompilerRTShared PRIVATE MODULAR_BUILDING_COMPILERRT)

# A fully-staticized version of KGENCompilerRT that we can use for standalone
# mojo executables.
add_modular_library(KGENCompilerRT-static
  ${SOURCES}

  $<TARGET_OBJECTS:obj.MSupportGlobals-static>
  $<TARGET_OBJECTS:obj.LLCLRuntimeGlobals-static>
  $<TARGET_OBJECTS:obj.MSupport>
  $<TARGET_OBJECTS:obj.LLCLRuntime>
)
target_compile_definitions(KGENCompilerRT-static PRIVATE MODULAR_BUILDING_COMPILERRT MODULAR_NO_EXPORT)

# Copy to the binary directory so that `mojo` can find this easily at runtime.
add_custom_command(TARGET KGENCompilerRT-static POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:KGENCompilerRT-static> ${CMAKE_BINARY_DIR}/bin/
)
