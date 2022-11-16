##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_mlir_dialect(Cache cache)
add_modular_doc(Cache -gen-dialect-doc CacheDialect Support/docs -dialect=Cache)

set(LLVM_TARGET_DEFINITIONS Cache.td)
mlir_tablegen(CacheAttrs.h.inc -gen-attrdef-decls -dialect=Cache)
mlir_tablegen(CacheAttrs.cpp.inc -gen-attrdef-defs -dialect=Cache)
add_public_tablegen_target(MLIRCacheAttrsIncGen)
add_dependencies(MLIRCacheIncGen MLIRCacheAttrsIncGen)
