//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_INTERPRETER_INTERPRETERATTRS_H
#define KGEN_INTERPRETER_INTERPRETERATTRS_H

#include "KGEN/Interpreter/InterpreterDialect.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"

//===----------------------------------------------------------------------===//
// MemoryBlob
//===----------------------------------------------------------------------===//

namespace M {
enum class MemoryKind : uint8_t { Heap, Stack, ConstGlobal };

class MemoryBlob {
public:
  /// A pointer region is a chunk of memory in the reference blob that
  /// represents a pointer.
  struct PointerRegion {
    /// The location of the region within the current blob.
    int64_t offset;
    /// The index of the referenced blob.
    int64_t blobIndex;
    /// The offset into the reference blob.
    int64_t blobOffset;
  };

  /// Create a memory blob.
  MemoryBlob(MemoryHandle hdl, MemoryKind kind,
             SmallVector<PointerRegion> pointerRegions)
      : hdl(hdl), kind(kind), pointerRegions(std::move(pointerRegions)) {}

  /// Get the memory handle.
  MemoryHandle getHandle() const { return hdl; }
  /// Get the memory kind.
  MemoryKind getKind() const { return kind; }
  /// Get the pointer offsets.
  ArrayRef<PointerRegion> getPointerRegions() const { return pointerRegions; }

private:
  /// The handle to the dialect resource that contains the blob data.
  MemoryHandle hdl;
  /// The kind of memory.
  MemoryKind kind;
  /// The offsets into the data that represent pointers.
  SmallVector<PointerRegion> pointerRegions;
};
} // namespace M

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "KGEN/Interpreter/InterpreterAttrs.h.inc"

#endif // KGEN_INTERPRETER_INTERPRETERATTRS_H
