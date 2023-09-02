//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef INTERPRETERATTRS_TD
#define INTERPRETERATTRS_TD

include "KGEN/Interpreter/InterpreterDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

class InterpreterAttr<string attrName, string attrMnemonic,
                      list<Trait> traits = []>
    : AttrDef<InterpreterDialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// PointerAttr
//===----------------------------------------------------------------------===//

def PointerAttr : InterpreterAttr<"Pointer", "pointer",
    [TypedAttrInterface]> {
  let summary = "an opaque pointer attribute";
  let description = [{
    The `#interp.pointer` attribute represents an opaque pointer value as an
    address. Pointer constants have no meaning outside of in-memory IR, so the
    textual format of this attribute has no inherent meaning. It is meant to be
    used in operation folders or the interpreter.
  }];

  let parameters = (ins "int64_t":$addr, AttributeSelfTypeParameter<"">:$type);
  let assemblyFormat = "`<` $addr `>`";

  let builders = [
    AttrBuilderWithInferredContext<(ins "int64_t":$addr, "Type":$type), [{
      return get(type.getContext(), addr, type);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// StoreToMemAttr
//===----------------------------------------------------------------------===//

def StoreToMemAttr : InterpreterAttr<"StoreToMem", "store_to_mem",
    [TypedAttrInterface]> {
  let summary = "wrap an attribute in stack memory";
  let description = [{
    The `#interp.store_to_mem` attribute indicates to the interpreter that the
    value should be stored into memory before used.

    Example:

    ```mlir
    #interp.store_to_mem<5 : index> : !M.pointer<index>
    ```
  }];

  let parameters = (ins "TypedAttr":$value,
                        AttributeSelfTypeParameter<"">:$type);
  let assemblyFormat = "`<` $value `>`";

  let builders = [
    AttrBuilderWithInferredContext<(ins "TypedAttr":$value, "Type":$type), [{
      return $_get(value.getContext(), value, type);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// MemorySpaceAttr
//===----------------------------------------------------------------------===//

def MemorySpaceAttr : ArrayOfAttr<InterpreterDialect, "MemorySpace",
                                  "memory_space", "MemoryBlob"> {
  let summary = "IR reprsentation of a memory space and values";
  let description = [{
    The `#interp.memory_space` attribute describes an abstract memory space
    consisting of distinct memory allocations represented as blobs. The contents
    of the blobs can be either dumb bits or pointers. Pointers are specially
    indicated and encoded, consisting of the blob index and the offset.

    Example:

    ```mlir
    #interp.memory_space<[
      (blob0, stack, [(24, 1, 8), (50, 0, 0), (64, 1, 8))]),
      (blob1, heap, [(0, 1, 8), (8, 1, 16)])
    ]>
    ```
  }];

  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// MemRefAttr
//===----------------------------------------------------------------------===//

def MemRefAttr : InterpreterAttr<"MemRef", "memref", [TypedAttrInterface]> {
  let summary = "a reference to a block of memory";
  let description = [{
    The `#interp.memref` attribute represents a reference to a block of memory. The
    block of memory lives in a dialect resource and describes the contents of
    the memory, and this attribute describes an offset into the block, which
    may be out-of-bounds, and how to interpret the memory.

    TODO: Add address space and write protection.

    Example:

    ```mlir
    #interp.memref<my_blob, 24> : !M.pointer<index>
    ```
  }];

  let parameters = (ins "MemorySpaceAttr":$memory, "int64_t":$index,
                        "int64_t":$offset,
                        AttributeSelfTypeParameter<"">:$type);
  let assemblyFormat = [{
    `<` $memory `,` $index `,` $offset `>`
  }];

  let builders = [
    AttrBuilderWithInferredContext<(ins "MemorySpaceAttr":$memory,
                                        "int64_t":$index, "int64_t":$offset,
                                        "Type":$type), [{
      return $_get(type.getContext(), memory, index, offset, type);
    }]>
  ];

  let genVerifyDecl = 1;
}

#endif // INTERPRETERATTRS_TD
