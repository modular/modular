//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef CACHEOPS_TD
#define CACHEOPS_TD

include "Support/CacheDialect/CacheDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

/// Base class for all Cache operations.
class CacheOp<string mnemonic, list<Trait> traits = []> :
    Op<CacheDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Cache_SymbolOp
//===----------------------------------------------------------------------===//


def Cache_SymbolOp
    : CacheOp<"symbol", [Symbol,
                         DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "cached symbol op";
  let description = [{
    This operation provides a proxy for a symbol that's been cached. It defines
    a symbol in a scope while eliding the body of that symbol. It stores the
    kind of the cached operation along with the original attrs in order to
    make it simple to roundtrip between a cached operation and the operation
    itself.

    This operation can take arbitrary arguments and produce arbitrary results.
    It also has a type to maintain the type information of the object that's
    being cached. This means that many analyses will still function properly
    since the symbols still exist and their types are still correct.

    For example:
    ```mlir
    cache.symbol @foo "func.func"
      regions=[#cache.region_hash<"HASHREGION0"[@callee]>]
      original_attrs={anAttr=#dialect.attr}
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$operands,
                       StrAttr:$sym_name,
                       StrAttr:$cachedOperation,
                       Cache_RegionHashArrayAttr:$regionHashes,
                       DictionaryAttr:$origAttrs);
  let results = (outs Variadic<AnyType>:$results);

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Cache_SymbolOp
//===----------------------------------------------------------------------===//

def Cache_ContainerOp : CacheOp<"container"> {
  let summary = "dummy op to hang regions off of";
  let description = [{
    This op provides an op we can serialize to hang regions off of. It can be
    printed for debugging, but it should not appear in the IR very often, if
    ever.
  }];

  let regions = (region SizedRegion<1>:$bodyRegion);

  let builders = [
    // Create a container with the given region as the body. It takes the body
    // of `body`.
    OpBuilder<(ins "Region &":$body)>
  ];

  let assemblyFormat = "attr-dict-with-keyword $bodyRegion";
}

#endif // CACHEOPS_TD
