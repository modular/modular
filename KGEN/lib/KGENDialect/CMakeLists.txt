##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

if (MODULAR_PARANOID)
  add_compile_definitions(MODULAR_PARANOID)
endif()

if (MODULAR_RELEASE_PACKAGE_BUILD)
  add_compile_definitions(MODULAR_RELEASE_PACKAGE_BUILD)
endif()

string(TOLOWER ${CMAKE_BUILD_TYPE} build_type)
string(TOLOWER ${MODULAR_KERNELS_BUILD_TYPE} kernels_build_type)
add_compile_definitions(BUILD_TYPE=${build_type})
add_compile_definitions(KERNELS_BUILD_TYPE=${kernels_build_type})

add_compile_definitions(
  MODULAR_LLCL_MAX_PROFILING_LEVEL=${MODULAR_LLCL_MAX_PROFILING_LEVEL})

add_modular_dialect_library(KGENDialect
  KGENAttrInterfaces.cpp
  KGENAttrs.cpp
  KGENDialect.cpp
  KGENInterfaces.cpp
  KGENOps.cpp
  KGENOpsFolders.cpp
  KGENParameters.cpp
  KGENTypeInterfaces.cpp
  KGENTypes.cpp
  KGENUtils.cpp
  ParameterEvaluator.cpp

  ADDITIONAL_HEADER_DIRS
  ${MODULAR_MAIN_INCLUDE_DIR}/KGEN/KGENDialect

  DEPENDS
  MLIRKGENIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  DebugInfoDialect
  HLCFDialect
  InterpreterInterface
  MCompilerSupport
  MSupport
  MSupportML
  MProfiling
  MLIRIR
  MLIRFunctionInterfaces
)
