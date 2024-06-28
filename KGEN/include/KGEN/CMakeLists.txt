##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_subdirectory(MOGGPreElab)
add_subdirectory(KGENVersion)
add_subdirectory(Interpreter)
add_subdirectory(HLCFDialect)
add_subdirectory(KGENDialect)
add_subdirectory(LITDialect)
add_subdirectory(POPDialect)
add_subdirectory(CODialect)
add_subdirectory(CustomDialect)

add_modular_passes(MLIRKGENPasses KGENPasses KGEN/docs/)
