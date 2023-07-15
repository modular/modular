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

# The LLVM libraries generally aren't OBJECT libraries, but we need
# KGENCompilerRT-static to be fully self contained, so we need to extract the
# object files from the LLVM libraries and link them into KGENCompilerRT-static.
get_property(LLVMSupportTargetFile TARGET LLVMSupport PROPERTY LOCATION)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${LLVMSupportTargetFile})
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/llvmsupport)
execute_process(
    COMMAND ${LLVM_BUILD_DIR}/bin/llvm-ar${CMAKE_EXECUTABLE_SUFFIX} -x ${LLVMSupportTargetFile}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/llvmsupport
)
file(GLOB LLVMSupportObjs ${CMAKE_CURRENT_BINARY_DIR}/llvmsupport/*)

# A fully-staticized version of KGENCompilerRT that we can use for standalone
# mojo executables.
add_modular_library(KGENCompilerRT-static
  ${SOURCES}

  $<TARGET_OBJECTS:obj.MSupportGlobals-static>
  $<TARGET_OBJECTS:obj.LLCLRuntimeGlobals-static>
  $<TARGET_OBJECTS:obj.MSupport>
  $<TARGET_OBJECTS:obj.LLCLRuntime>
  ${LLVMSupportObjs}
)
target_compile_definitions(KGENCompilerRT-static PRIVATE MODULAR_BUILDING_COMPILERRT MODULAR_NO_EXPORT)

# Copy to the binary directory so that `mojo` can find this easily at runtime.
add_custom_command(TARGET KGENCompilerRT-static POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:KGENCompilerRT-static> ${CMAKE_BINARY_DIR}/bin/
)
