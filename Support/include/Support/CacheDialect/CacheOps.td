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
// Cache_ContainerOp
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
