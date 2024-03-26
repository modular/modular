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
enum class MemoryKind : uint8_t { Heap, Stack, ConstGlobal, Persistent };

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
} // namespace M

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "KGEN/Interpreter/InterpreterAttrs.h.inc"

#endif // KGEN_INTERPRETER_INTERPRETERATTRS_H
