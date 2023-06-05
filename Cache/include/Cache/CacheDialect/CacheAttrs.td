//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef CACHEATTRS_TD
#define CACHEATTRS_TD

include "Cache/CacheDialect/CacheDialect.td"
include "Cache/CacheDialect/CacheAttrInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

class CacheAttr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<CacheDialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// ConstantHashAttr
//===----------------------------------------------------------------------===//

def Cache_ConstantHashAttr : CacheAttr<"ConstantHash", "constant_hash",
    [ElementsAttrInterface, TypedAttrInterface,
     DeclareAttrInterfaceMethods<ReplaceableAttrInterface>
     ]> {
  let summary = "A non-parametrized hash.";
  let description = [{
    This attribute holds a simple hash - it cannot be parametrized. This is
    suggestively useful for constants or other large binary blobs that need
    to be stored in the cache but will never be parametrized. Note that the
    intended purpose of this attribute is to store things that are conceptually
    "outside the IR". Even in the case of storing object code inline, the idea
    is that changing out the constant behind a ConstantHashAttr could change
    the output of the program, but does not change the program itself.

    Using the constant behind this hash requires materialization/reinflation.

    For example:
    ```mlir
    /// No alignment necessary
    %0 = arith.constant #cache.constant_hash<"HASHWEIGHT"> : tensor<300x4096xf32>
    /// With some alignment.
    %1 = arith.constant #cache.constant_hash<"HASHWEIGHT", {align = 4096 : ui64}> : tensor<300x4096xf32>
    ```
  }];

  let parameters = (ins AttributeSelfTypeParameter<"", "ShapedType">:$type,
                        StringRefParameter<"">:$hash,
                        "DictionaryAttr":$additionalData);
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    /// Returns any 'align' entry in the attribute's additionalData. Otherwise
    /// returns nullopt.
    std::optional<uint64_t> getOptAlign() const;
  }];
}

//===----------------------------------------------------------------------===//
// RegionHashAttr
//===----------------------------------------------------------------------===//

def Cache_RegionHashAttr : CacheAttr<"RegionHash", "region_hash"> {
  let summary = "A parametrized hash of an MLIR region.";
  let description = [{
    This attribute holds a hash of an MLIR region parametrized by the special
    attributes contained within that region. Conceptually, this is a
    representation of a program, or a representation of things that are
    "inside this IR". Changing a RegionHashAttr is equivalent to changing the
    program, which is why the RegionHashAttr doesn't implement the
    ReplaceableAttrInterface.

    For example:
    ```mlir
    /// When the region refers to symbols:
    #cache.region_hash<"REGIONHASH", params=[@callee1, @callee2]>

    /// When the region contains other hashes:
    #cache.region_hash<"REGIONHASH",
                       params=[@callee1, @callee2, "HASHCONSTANT"]>

    /// When the region has no parameter references:
    #cache.region_hash<"REGIONHASH">
    ```
  }];

  let parameters = (ins StringRefParameter<"">:$hash,
                        OptionalArrayRefParameter<"Attribute">:$params);
  let assemblyFormat = [{
    `<` custom<Base64>($hash) (` ` `params` `=` `[` $params^ `]`)? `>`
  }];
}

/// An array of RegionHashAttrs.
def Cache_RegionHashArrayAttr :
    ArrayOfAttr<CacheDialect, "RegionHashArray", "regions", "RegionHashAttr">;

//===----------------------------------------------------------------------===//
// HashIndexAttr
//===----------------------------------------------------------------------===//

def Cache_HashIndexAttr : CacheAttr<"HashIndex", "hash_index",
    [DeclareAttrInterfaceMethods<ReplaceableAttrIndexInterface>]> {
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

  let parameters = (ins "uint64_t":$index);
  let assemblyFormat = "`<` $index `>`";
}

//===----------------------------------------------------------------------===//
// TypedHashIndexAttr
//===----------------------------------------------------------------------===//

def Cache_TypedHashIndexAttr
  : CacheAttr<"TypedHashIndex", "typed_hash_index", [TypedAttrInterface,
    DeclareAttrInterfaceMethods<ReplaceableAttrIndexInterface>]> {
  let summary = "A typed hash reference.";
  let description = [{
    This attribute is a index into the list of hashes on the RegionHashAttr.
    It's used to reference one of the hashes in that list by a replacement for
    constant-like ops in order to orthogonalize the cache and the constants
    stored in the graph. This version is typed, for use when replacing typed
    attributes.

    For example:
    ```mlir
    /// Operations won't necessarily verify when this attribute is used!
    cache.constant {value=#cache.typed_hash_index<1234> : i32}
    ```
  }];

   let parameters = (ins AttributeSelfTypeParameter<"", "Type">:$type,
                         "uint64_t":$index);

   let assemblyFormat = "`<` $index `>`";
}

#endif // CACHEATTRS_TD
