##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_mlir_dialect(Cache cache -dialect=cache)
add_modular_doc(Cache -gen-dialect-doc CacheDialect Support/docs -dialect=cache)

set(LLVM_TARGET_DEFINITIONS Cache.td)
mlir_tablegen(CacheAttrs.h.inc -gen-attrdef-decls -dialect=Cache)
mlir_tablegen(CacheAttrs.cpp.inc -gen-attrdef-defs -dialect=Cache)
add_public_tablegen_target(MLIRCacheAttrsIncGen)
add_dependencies(MLIRCacheIncGen MLIRCacheAttrsIncGen)

set(LLVM_TARGET_DEFINITIONS CacheAttrInterfaces.td)
mlir_tablegen(CacheAttrInterfaces.h.inc -gen-attr-interface-decls)
mlir_tablegen(CacheAttrInterfaces.cpp.inc -gen-attr-interface-defs)
add_public_tablegen_target(MLIRCacheAttrInterfacesIncGen)
add_dependencies(MLIRCacheIncGen MLIRCacheAttrInterfacesIncGen)
