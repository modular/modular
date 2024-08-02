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
include "mlir/IR/OpAsmInterface.td"

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
// SymbolicPointerAttr
//===----------------------------------------------------------------------===//

def SymbolicPointerAttr : InterpreterAttr<"SymbolicPointer", "symbolic_pointer",
    [TypedAttrInterface]> {
  let summary = "a pointer to symbolic interpreter memory";
  let description = [{
    The `#interp.symbolic_pointer` attribute represents a reference to a piece
    of symbolic interpreter memory. Symbolic memory has no runtime
    representation and consequently does not require a target model to use.
    However, bit-level manipulations are not supported because of the missing
    runtime representation. Symbolic memory can be also be used for
    non-materializable values inside the interpreter.

    Example:

    ```mlir
    #interp.symbolic_pointer<3>
    ```
  }];

  let parameters = (ins "uint64_t":$slot, AttributeSelfTypeParameter<"">:$type);
  let assemblyFormat = "`<` $slot `>`";

  let builders = [
    AttrBuilderWithInferredContext<(ins "uint64_t":$slot, "Type":$type), [{
      return get(type.getContext(), slot, type);
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
// MemoryHandleAttr
//===----------------------------------------------------------------------===//

def AlignedBlob : AttrParameter<"AlignedBlob", "an aligned byte array"> {
  let allocator = [{
    {
      size_t blobSize = $_self.data.size();
      void *blobMemory = $_allocator.allocate(blobSize, $_self.align);
      memcpy(blobMemory, $_self.data.data(), blobSize);
      ArrayRef<char> data((const char *)blobMemory, blobSize);
      $_dst = AlignedBlob($_self.align, data, $_self.isString);
    }
  }];
  let cppStorageType = "OwnedAlignedBlob";
}

def MemoryHandleAttr : InterpreterAttr<"MemoryHandle", "memory_handle"> {
  let summary = "reference to a physical blob of memory";
  let description = [{
    The `#interp.memory_handle` attribute contains an array of aligned bytes. It
    is the backing memory of materialized interpreter memory blobs.
  }];

  let parameters = (ins AlignedBlob:$memory);
  let assemblyFormat = "`<` custom<AlignedHex>($memory) `>`";

  let builders = [
    AttrBuilder<(ins "uint64_t":$align, "ArrayRef<char>":$memory), [{
      return $_get($_ctxt, AlignedBlob{align, memory, /*isString=*/false});
    }]>,
    AttrBuilder<(ins "StringRef":$str)>
  ];

  let extraClassDeclaration = [{
    const char *getData() { return getMemory().data.data(); }
    size_t getSize() {  return getMemory().data.size(); }
    uint64_t getAlign() { return getMemory().align; }
    bool isString() { return getMemory().isString; }
  }];
}

//===----------------------------------------------------------------------===//
// MemoryBlobAttr
//===----------------------------------------------------------------------===//

def MemoryBlobAttr : InterpreterAttr<"MemoryBlob", "memory_blob"> {
  let summary = "represents a single interpreter memory blob";
  let description = [{
    The `#interp.memory_blob` attribute represents a single interpreter memory
    blob, which backs a single allocation. It contains a reference to the
    underlying resource that contains the memory value, the kind of allocation,
    and the tracked pointer regions inside the blob.
  }];

  let parameters = (ins
    "MemoryHandleAttr":$handle,
    "MemoryKind":$kind,
    ArrayRefParameter<"PointerRegion">:$pointerRegions
  );

  let builders = [
    AttrBuilderWithInferredContext<(ins
        "MemoryHandleAttr":$handle,
        "MemoryKind":$kind,
        "ArrayRef<PointerRegion>":$pointerRegions), [{
      return $_get(handle.getContext(), handle, kind, pointerRegions);
    }]>
  ];

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// MemorySpaceAttr
//===----------------------------------------------------------------------===//

def MemorySpaceAttr : ArrayOfAttr<InterpreterDialect, "MemorySpace",
                                  "memory_space", "MemoryBlobAttr"> {
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
