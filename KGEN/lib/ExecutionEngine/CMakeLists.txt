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

add_modular_library(KGENExecutionEngine
  ExecutionEngine.cpp

  LINK_COMPONENTS
  OrcJIT
  OrcShared
  OrcTargetProcess
  JITLink
  MC

  LINK_LIBS PUBLIC
  LLCLRuntime
  MCache
  MSupport
  )

if (NOT MODULAR_BUILD_ARCH)
  cache_files(orc-rt-cas
  BASE_DIR ${MODULAR_DERIVED_DIR}/.kgen_cache/orc
  HEADER_NAME ${MODULAR_BINARY_DIR}/KGEN/include/KGEN/ExecutionEngine/ORCCASID.h
  INPUTS ${ORC_LIB}
  INPUT_NAMES "kOrcRT"
  RELATIVE_INSTALL_DIR ".kgen_cache/orc")

  cache_files(compiler-rt-cas
    BASE_DIR ${MODULAR_DERIVED_DIR}/.kgen_cache/orc
    HEADER_NAME ${MODULAR_BINARY_DIR}/KGEN/include/KGEN/ExecutionEngine/COMPILERRTCASID.h
    INPUTS $<TARGET_FILE:KGENCompilerRTShared>
    INPUT_NAMES "kCompilerRT"
    RELATIVE_INSTALL_DIR ".kgen_cache/orc"
  )

  add_dependencies(KGENExecutionEngine orc-rt-cas compiler-rt-cas)
endif()
