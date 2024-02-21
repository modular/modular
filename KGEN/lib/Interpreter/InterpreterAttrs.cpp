//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/Interpreter/InterpreterAttrs.h"
#include "KGEN/Interpreter/InterpreterDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;

//===----------------------------------------------------------------------===//
// InterpreterDialect
//===----------------------------------------------------------------------===//

void InterpreterDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "KGEN/Interpreter/InterpreterAttrs.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// MemorySpaceAttr
//===----------------------------------------------------------------------===//

namespace mlir {
template <>
struct FieldParser<MemoryBlob> {
  static FailureOr<MemoryBlob> parse(AsmParser &p) {
    if (p.parseLParen())
      return failure();
    FailureOr<MemoryHandle> hdl = p.parseResourceHandle<MemoryHandle>();
    if (failed(hdl))
      return failure();
    StringRef kindStr;
    SmallVector<MemoryBlob::PointerRegion> pointerRegions;
    if (p.parseComma() || p.parseKeyword(&kindStr) || p.parseComma() ||
        p.parseCommaSeparatedList(
            AsmParser::Delimiter::Square,
            [&] {
              MemoryBlob::PointerRegion &region = pointerRegions.emplace_back();
              return failure(
                  p.parseLParen() || p.parseInteger(region.offset) ||
                  p.parseComma() || p.parseInteger(region.blobIndex) ||
                  p.parseComma() || p.parseInteger(region.blobOffset) ||
                  p.parseRParen());
            }) ||
        p.parseRParen())
      return failure();
    MemoryKind kind = llvm::StringSwitch<MemoryKind>(kindStr)
                          .Case("stack", MemoryKind::Stack)
                          .Case("heap", MemoryKind::Heap)
                          .Case("const_global", MemoryKind::ConstGlobal)
                          .Case("persistent", MemoryKind::Persistent);
    return MemoryBlob(*hdl, kind, std::move(pointerRegions));
  }
};

static AsmPrinter &operator<<(AsmPrinter &p, const MemoryBlob &blob) {
  p << '(';
  p.printResourceHandle(blob.getHandle());
  p << ", ";
  switch (blob.getKind()) {
  case MemoryKind::Stack:
    p << "stack";
    break;
  case MemoryKind::Heap:
    p << "heap";
    break;
  case MemoryKind::ConstGlobal:
    p << "const_global";
    break;
  case MemoryKind::Persistent:
    p << "persistent";
    break;
  }
  p << ", [";
  llvm::interleaveComma(blob.getPointerRegions(), p,
                        [&](const MemoryBlob::PointerRegion &region) {
                          p << '(' << region.offset << ", " << region.blobIndex
                            << ", " << region.blobOffset << ')';
                        });
  p << "])";
  return p;
}
} // namespace mlir

namespace M {
static llvm::hash_code hash_value(const MemoryBlob &b) {
  return llvm::hash_combine(b.getHandle(), b.getKind(), b.getPointerRegions());
}

static bool operator==(const MemoryBlob &lhs, const MemoryBlob &rhs) {
  return std::make_tuple(lhs.getHandle(), lhs.getKind(),
                         lhs.getPointerRegions()) ==
         std::make_tuple(rhs.getHandle(), rhs.getKind(),
                         rhs.getPointerRegions());
}

static llvm::hash_code hash_value(const MemoryBlob::PointerRegion &region) {
  return llvm::hash_combine(region.offset, region.blobIndex, region.blobOffset);
}

static bool operator==(const MemoryBlob::PointerRegion &lhs,
                       const MemoryBlob::PointerRegion &rhs) {
  return std::make_tuple(lhs.offset, lhs.blobIndex, lhs.blobOffset) ==
         std::make_tuple(rhs.offset, rhs.blobIndex, rhs.blobOffset);
}
} // namespace M

LogicalResult
MemorySpaceAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                        ArrayRef<MemoryBlob> blobs) {
  for (auto [i, blob] : llvm::enumerate(blobs)) {
    if (blob.getKind() == MemoryKind::ConstGlobal &&
        !blob.getPointerRegions().empty()) {
      return emitError() << "const_global blob #" << i
                         << " cannot have pointer regions";
    }
    for (const MemoryBlob::PointerRegion &region : blob.getPointerRegions()) {
      if (region.blobIndex < 0 ||
          static_cast<size_t>(region.blobIndex) >= blobs.size()) {
        return emitError() << "blob #" << i << " pointer at offset "
                           << region.offset
                           << " has an out-of-bounds blob index: "
                           << region.blobIndex;
      }
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// MemRefAttr
//===----------------------------------------------------------------------===//

LogicalResult MemRefAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                 MemorySpaceAttr space, int64_t index,
                                 int64_t offset, Type type) {
  if (index < 0 || static_cast<size_t>(index) >= space.size())
    return emitError() << "memref blob index " << index << " is out-of-bounds";
  return success();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "KGEN/Interpreter/InterpreterAttrs.cpp.inc"
