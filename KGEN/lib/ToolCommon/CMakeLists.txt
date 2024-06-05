##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_subdirectory(InitAllDialects)
add_subdirectory(TranslationRegistry)
add_subdirectory(CompilationOptions)

add_modular_library(KGENToolCommon
  CLOptions.cpp
  PassRegistry.cpp
  PassManagerConfigOptions.cpp

  LINK_LIBS PUBLIC
  KGENInitAllDialects
  KGENTranslationRegistry
  KGENCompilationOptions

  LowerLIT
  KGENToLLVM
  KGENElaborator
  KGENTransforms
  DebugInfoToLLVM
  MOGGPreElabPasses

  MLIRIndexDialect
  MLIRLLVMDialect
  MLIRBuiltinToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
  MLIRNVVMToLLVMIRTranslation
  )
