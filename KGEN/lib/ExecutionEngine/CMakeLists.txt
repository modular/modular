##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

set(LLVMCompilerRTLibPath ${LLVM_BUILD_DIR}/lib/clang/${LLVM_VERSION_MAJOR})
if (APPLE)
  set(ORCSuffix "_osx")
  set(LLVMCompilerRTLibPath ${LLVMCompilerRTLibPath}/lib/darwin)
elseif (WIN32)
  set(ORCSuffix "-x86_64")
  set(LLVMCompilerRTLibPath ${LLVMCompilerRTLibPath}/lib/windows)
elseif (UNIX)
  set(ORCSuffix "")

  # Get the architecture.
  set(arch ${CMAKE_HOST_SYSTEM_PROCESSOR})
  if (arch STREQUAL "x86")
    set(arch "i686")
  endif ()
  # Get the vendor.
  set(vendor "unknown")
  # Get os.
  string(TOLOWER ${CMAKE_HOST_SYSTEM_NAME} os)
  set(triple "${arch}-${vendor}-${os}-gnu")
  message(STATUS "Host triple: ${triple}")
  set(LLVMCompilerRTLibPath ${LLVMCompilerRTLibPath}/lib/${triple})
endif ()

list(APPEND CMAKE_PREFIX_PATH ${LLVMCompilerRTLibPath})

find_library(ORC_LIB orc_rt${ORCSuffix} REQUIRED
  HINTS "${MODULAR_SOURCE_DIR}/${LLVMCompilerRTLibPath}")

# Generate the KGEN version string so we can use it for the cache_files call.
include(${CMAKE_SOURCE_DIR}/KGEN/include/KGEN/KGENVersion/GenerateVersion.cmake)
set(VERSION_STRING "${MODULAR_VERSION_MAJOR}.${MODULAR_VERSION_MINOR}.${MODULAR_VERSION_PATCH}${MODULAR_VERSION_LABEL}-${KGEN_VERSION_REVISION}-${MODULAR_BUILD_TYPE_LOWER}")

cache_files(orc-rt-cas
  BASE_DIR ${MODULAR_DERIVED_DIR}/.mojo_cache/orc
  HEADER_NAME ${MODULAR_BINARY_DIR}/KGEN/include/KGEN/ExecutionEngine/ORCCASID.h
  HEADER_RELATIVE_NAME KGEN/ExecutionEngine/ORCCASID.h
  INPUTS ${ORC_LIB}
  VERSION ${VERSION_STRING}
  INPUT_NAMES "kOrcRT"
  RELATIVE_INSTALL_DIR ".mojo_cache/orc"
)

cache_files(compiler-rt-cas
  BASE_DIR ${MODULAR_DERIVED_DIR}/.mojo_cache/orc
  HEADER_NAME ${MODULAR_BINARY_DIR}/KGEN/include/KGEN/ExecutionEngine/COMPILERRTCASID.h
  HEADER_RELATIVE_NAME KGEN/ExecutionEngine/COMPILERRTCASID.h
  INPUTS $<TARGET_FILE:KGENCompilerRTShared>
  VERSION ${VERSION_STRING}
  INPUT_NAMES "kCompilerRT"
  RELATIVE_INSTALL_DIR ".mojo_cache/orc"
)

set(CAS_DEPENDS compiler-rt-cas orc-rt-cas)

add_modular_library(KGENExecutionEngine
  ExecutionEngine.cpp

  DEPENDS
  ${CAS_DEPENDS}
  KGENVersion

  LINK_COMPONENTS
  OrcDebugging
  OrcJIT
  OrcShared
  OrcTargetProcess
  JITLink
  MC

  LINK_LIBS PUBLIC
  KGENSupport
  LLCLRuntime
  MCache
  MSupport
  LLVMPerfJITEvents
  LLVMOrcTargetProcess
)
set_property(TARGET KGENExecutionEngine PROPERTY KGEN_ORC_RT_LIB "${ORC_LIB}")
target_compile_definitions(KGENExecutionEngine PRIVATE KGEN_ASAN_LIB="${KGEN_ASAN_LIBRARY}")
target_compile_definitions(KGENExecutionEngine PRIVATE KGEN_TSAN_LIB="${KGEN_TSAN_LIBRARY}")
