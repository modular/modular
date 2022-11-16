//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef CACHEATTRS_TD
#define CACHEATTRS_TD

include "Support/CacheDialect/CacheDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/SubElementInterfaces.td"

class CacheAttr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<CacheDialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// RegionHashAttr
//===----------------------------------------------------------------------===//

def SymbolListParameter : OptionalArrayRefParameter<"FlatSymbolRefAttr",
    "symbols in region">;

def Cache_RegionHashAttr : CacheAttr<"RegionHash", "region_hash",
    [DeclareAttrInterfaceMethods<SubElementAttrInterface>]> {
  let summary = "A parametrized hash of an MLIR region.";
  let description = [{
    This attribute holds a hash of an MLIR region parametrized by the symbols
    referenced inside that region.

    For example:
    ```mlir
    /// When the region refers to symbols:
    #cache.region_hash<"REGIONHASH"[@callee1, @callee2]>

    /// When the region has no symbol references:
    #cache.region_hash<"REGIONHASH">
    ```
  }];

  let parameters = (ins StringRefParameter<"">:$hash,
                        SymbolListParameter:$symbols);
  let assemblyFormat = [{
    `<` custom<Base64>($hash) (`[` $symbols^ `]`)? `>`
  }];
}

//===----------------------------------------------------------------------===//
// SymbolIndexAttr
//===----------------------------------------------------------------------===//

def Cache_SymbolIndexAttr : CacheAttr<"SymbolIndex", "symbol_index",
    [DeclareAttrInterfaceMethods<SubElementAttrInterface>]> {
  let summary = "A symbol reference.";
  let description = [{
    This attribute is a index into the list of symbols on the RegionHashAttr.
    It's used to reference one of the symbols in that list by a replacement for
    call-like ops in order to orthogonalize the cache and the call graph.

    For example:
    ```mlir
    /// Operations won't necessarily verify when this attribute is used!
    func.call {callee=#cache.symbol_index<1234>}
    ```
  }];

  let parameters = (ins "uint64_t":$index);
  let assemblyFormat = "`<` $index `>`";
}

#endif // CACHEATTRS_TD
