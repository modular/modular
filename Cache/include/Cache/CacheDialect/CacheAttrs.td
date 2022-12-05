//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef CACHEATTRS_TD
#define CACHEATTRS_TD

include "Cache/CacheDialect/CacheDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/SubElementInterfaces.td"

class CacheAttr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<CacheDialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// ConstantHashAttr
//===----------------------------------------------------------------------===//

def Cache_ConstantHashAttr : CacheAttr<"ConstantHash", "constant_hash",
    [ElementsAttrInterface, TypedAttrInterface]> {
  let summary = "A non-parametrized hash.";
  let description = [{
    This attribute holds a simple hash - it cannot be parametrized. This is
    suggestively useful for constants or other large binary blobs that need
    to be stored in the cache but will never be parametrized.

    Using the constant behind this hash requires materialization.

    For example:
    ```mlir
    /// No alignment necessary
    %0 = arith.constant #cache.constant_hash<"HASHWEIGHT"> : tensor<300x4096xf32>
    /// With some alignment.
    %1 = arith.constant #cache.constant_hash<"HASHWEIGHT", {align = 4096 : ui64}> : tensor<300x4096xf32>
    ```
  }];

  let parameters = (ins AttributeSelfTypeParameter<"", "Type">:$type,
                        StringRefParameter<"">:$hash,
                        "DictionaryAttr":$additionalData);
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// RegionHashAttr
//===----------------------------------------------------------------------===//

def SymbolListParameter : OptionalArrayRefParameter<"SymbolRefAttr",
    "symbols in region">;

def ConstantHashListParameter : OptionalArrayRefParameter<"ConstantHashAttr",
    "hashes in region">;

def Cache_RegionHashAttr : CacheAttr<"RegionHash", "region_hash",
    [DeclareAttrInterfaceMethods<SubElementAttrInterface>]> {
  let summary = "A parametrized hash of an MLIR region.";
  let description = [{
    This attribute holds a hash of an MLIR region parametrized by the symbols
    referenced inside that region.

    For example:
    ```mlir
    /// When the region refers to symbols:
    #cache.region_hash<"REGIONHASH", symbols=[@callee1, @callee2]>

    /// When the region contains other hashes:
    #cache.region_hash<"REGIONHASH", symbols=[@callee1, @callee2],
                       hashes=["HASHCONSTANT"]>

    /// When the region has no symbol references:
    #cache.region_hash<"REGIONHASH">
    ```
  }];

  let parameters = (ins StringRefParameter<"">:$hash,
                        SymbolListParameter:$symbols,
                        ConstantHashListParameter:$hashes);
  let assemblyFormat = [{
    `<` custom<Base64>($hash)
        (` ` `symbols` `=` `[` $symbols^ `]`)?
        (` ` `hashes` `=` `[` $hashes^ `]`)? `>`
  }];
}

/// An array of RegionHashAttrs.
def Cache_RegionHashArrayAttr :
    ArrayOfAttr<CacheDialect, "RegionHashArray", "regions", "RegionHashAttr",
                [SubElementAttrInterface]>;

//===----------------------------------------------------------------------===//
// ParametrizedIndexAttrBase
//===----------------------------------------------------------------------===//

class Cache_ParametrizedIndexAttrBase<string attrName, string mnemonic>
  : CacheAttr<attrName, mnemonic,
    [DeclareAttrInterfaceMethods<SubElementAttrInterface>]> {
  let parameters = (ins "uint64_t":$index);
  let assemblyFormat = "`<` $index `>`";
}

//===----------------------------------------------------------------------===//
// HashIndexAttr
//===----------------------------------------------------------------------===//

def Cache_HashIndexAttr
  : Cache_ParametrizedIndexAttrBase<"HashIndex", "hash_index"> {
  let summary = "A hash reference.";
  let description = [{
    This attribute is a index into the list of hashes on the RegionHashAttr.
    It's used to reference one of the hashes in that list by a replacement for
    constant-like ops in order to orthogonalize the cache and the constants
    stored in the graph.

    For example:
    ```mlir
    /// Operations won't necessarily verify when this attribute is used!
    cache.constant {value=#cache.hash_index<1234>}
    ```
  }];
}

#endif // CACHEATTRS_TD
