//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/Interpreter/InterpreterDialect.h"
#include "KGEN/Interpreter/InterpreterAttrs.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;

//===----------------------------------------------------------------------===//
// InterpreterDialectBytecodeInterface
//===----------------------------------------------------------------------===//

using mlir::DialectBytecodeReader;
using mlir::DialectBytecodeWriter;
using mlir::get;

static LogicalResult readMemoryBlobs(DialectBytecodeReader &reader,
                                     SmallVectorImpl<MemoryBlob> &blobs) {
  uint64_t size;
  if (failed(reader.readVarInt(size)))
    return failure();
  blobs.reserve(size);

  auto readPointerRegion = [&](MemoryBlob::PointerRegion &region) {
    int64_t offset, blobIndex, blobOffset;
    if (failed(reader.readSignedVarInt(offset)) ||
        failed(reader.readSignedVarInt(blobIndex)) ||
        failed(reader.readSignedVarInt(blobOffset)))
      return failure();
    return LogicalResult::success();
  };

  for (unsigned i = 0; i < size; ++i) {
    FailureOr<MemoryHandle> hdl = reader.readResourceHandle<MemoryHandle>();
    uint64_t kind;
    if (failed(hdl) || failed(reader.readVarInt(kind)))
      return failure();
    SmallVector<MemoryBlob::PointerRegion> regions;
    if (failed(reader.readList(regions, readPointerRegion)))
      return failure();
    blobs.emplace_back(*hdl, static_cast<MemoryKind>(kind), std::move(regions));
  }

  return success();
}

static void writeMemoryBlobs(DialectBytecodeWriter &writer,
                             ArrayRef<MemoryBlob> blobs) {
  writer.writeVarInt(blobs.size());

  auto writePointerRegion = [&](const MemoryBlob::PointerRegion &region) {
    writer.writeSignedVarInt(region.offset);
    writer.writeSignedVarInt(region.blobIndex);
    writer.writeSignedVarInt(region.blobOffset);
  };

  for (const MemoryBlob &blob : blobs) {
    writer.writeResourceHandle(blob.getHandle());
    writer.writeVarInt(static_cast<uint64_t>(blob.getKind()));
    writer.writeList(blob.getPointerRegions(), writePointerRegion);
  }
}

namespace {
#include "KGEN/Interpreter/InterpreterDialectBytecode.cpp.inc"

struct InterpreterDialectBytecodeInterface
    : public mlir::BytecodeDialectInterface {
  InterpreterDialectBytecodeInterface(Dialect *dialect)
      : BytecodeDialectInterface(dialect) {}

  Attribute readAttribute(DialectBytecodeReader &reader) const override {
    return ::readAttribute(getContext(), reader);
  }

  LogicalResult writeAttribute(Attribute attr,
                               DialectBytecodeWriter &writer) const override {
    return ::writeAttribute(attr, writer);
  }

  Type readType(DialectBytecodeReader &reader) const override {
    return ::readType(getContext(), reader);
  }

  LogicalResult writeType(Type type,
                          DialectBytecodeWriter &writer) const override {
    return ::writeType(type, writer);
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// InterpreterDialect
//===----------------------------------------------------------------------===//

void InterpreterDialect::initialize() {
  registerAttributes();
  addInterface<InterpreterDialectBytecodeInterface>();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "KGEN/Interpreter/InterpreterDialect.cpp.inc"
