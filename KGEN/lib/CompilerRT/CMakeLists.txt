##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

include(AddTCMalloc)

set(SOURCES
  Config.cpp
  Globals.cpp
  IO.cpp
  InitIntelAMX.cpp
  Initialize.cpp
  LLCL.cpp
  Memory.cpp
  Python.cpp
  Random.cpp
  Support.cpp
  System.cpp
  Tracing.cpp
)

# A statically built CompilerRT library.
add_modular_library(KGENCompilerRT
  ${SOURCES}

  LINK_LIBS PUBLIC
  ArraySupport
  CUDASupport
  AsyncRTRuntime
  AsyncRTRuntimeGlobals
  MProfiling
  MSupport
  MSupportML
  MThreading
  Runtime
)

target_compile_definitions(KGENCompilerRT PRIVATE MODULAR_BUILDING_COMPILERRT -DMAX_CONFIG_SECTION=${MAX_CONFIG_SECTION})

# A dynamically built CompilerRT library used when jitting.
add_modular_library(KGENCompilerRTShared SHARED
  ${SOURCES}

  LINK_LIBS PRIVATE
  ArraySupport
  CUDASupport
  CUDASupportGlobals
  AsyncRTRuntime
  AsyncRTRuntimeGlobals
  MProfiling
  MSupport
  MSupportGlobals
  MSupportML
  MThreading
  Runtime
)

target_compile_definitions(KGENCompilerRTShared PRIVATE MODULAR_BUILDING_COMPILERRT -DMAX_CONFIG_SECTION=${MAX_CONFIG_SECTION})

# The LLVM libraries generally aren't OBJECT libraries, but we need
# KGENCompilerRT-static to be fully self contained, so we need to extract the
# object files from the LLVM libraries and link them into KGENCompilerRT-static.
function(add_static_dep_lib target)
  get_property(target_file TARGET ${target} PROPERTY LOCATION)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${target_file})
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/static_deps)
  execute_process(
    COMMAND ${LLVM_BUILD_DIR}/bin/llvm-ar${CMAKE_EXECUTABLE_SUFFIX} -x ${target_file}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/static_deps
  )
endfunction()

if(${MODULAR_LLCL_MAX_PROFILING_LEVEL})
  # FIXME(20613): this dependency might need to go away to keep the runtime small.

  # We log HostMachineInfo with profiling information. This has a dependency
  # on clangBasic.
  add_static_dep_lib(clangBasic)
endif()
if (TCMALLOC_LIBRARY)
  add_static_dep_lib(libtcmalloc)
endif()
add_static_dep_lib(LLVMSupport)
add_static_dep_lib(LLVMDemangle)
add_static_dep_lib(LLVMTargetParser)
add_static_dep_lib(MLIRSupport)
file(GLOB StaticDepObjects ${CMAKE_CURRENT_BINARY_DIR}/static_deps/*)

# Collect the various system libraries required by llvm support that we need to
# link against. We can extract this using llvm-config.
execute_process(
  COMMAND ${LLVM_TOOLS_BINARY_DIR}/llvm-config${CMAKE_EXECUTABLE_SUFFIX} --libs support --system-libs
  OUTPUT_VARIABLE config_output
)
# The system libs are on the second line of the output, so we need to split it
# up. They are space separated, so we can just replace the spaces with commas.
string(REGEX MATCHALL "[^\n]+" config_output "${config_output}")
list(GET config_output 1 kgen_system_libs)
string(REPLACE " " "," kgen_system_libs "${kgen_system_libs}")

set(COMPILER_RT_OBJECT_SRCS
  $<TARGET_OBJECTS:obj.MSupportGlobals-static>
  $<TARGET_OBJECTS:obj.CUDASupportGlobals-static>
  $<TARGET_OBJECTS:obj.AsyncRTRuntimeGlobals-static>
  $<TARGET_OBJECTS:obj.ArraySupport>
  $<TARGET_OBJECTS:obj.MSupport>
  $<TARGET_OBJECTS:obj.MSupportML>
  $<TARGET_OBJECTS:obj.MThreading>
  $<TARGET_OBJECTS:obj.MProfiling>
  $<TARGET_OBJECTS:obj.MVersion>
  $<TARGET_OBJECTS:obj.AsyncRTRuntime>
  $<TARGET_OBJECTS:obj.AsyncRTSupport>
  $<TARGET_OBJECTS:obj.CUDASupport>
  $<TARGET_OBJECTS:obj.Runtime>
)

if(${MODULAR_LLCL_MAX_PROFILING_LEVEL})
  # For logging HostMachineInfo with profiling results.
  list(APPEND COMPILER_RT_OBJECT_SRCS $<TARGET_OBJECTS:obj.MArchTarget>)
endif()

# A fully-staticized version of KGENCompilerRT that we can use for standalone
# mojo executables.
add_modular_library(KGENCompilerRT-static
  ${SOURCES}

  ${COMPILER_RT_OBJECT_SRCS}

  ${StaticDepObjects}
  )
target_compile_definitions(KGENCompilerRT-static PRIVATE MODULAR_BUILDING_COMPILERRT MODULAR_NO_EXPORT -DMAX_CONFIG_SECTION=${MAX_CONFIG_SECTION})
set_property(TARGET KGENCompilerRT-static PROPERTY KGEN_SYSTEM_LIBS "${kgen_system_libs}")

# Copy to the binary directory so that `mojo` can find this easily at runtime.
add_custom_command(TARGET KGENCompilerRT-static POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:KGENCompilerRT-static> ${CMAKE_BINARY_DIR}/bin/
  )
