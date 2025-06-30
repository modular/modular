//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/HashUtils.h"

#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BlockSupport.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

#include "xxh3.h"
#include "xxhash.h"

using namespace mlir;

namespace M {

/// A raw_ostream that hash the content using the xxhash algorithm.
class raw_xxhash_stream : public raw_ostream {
  XXH3_state_t State;

  /// See raw_ostream::write_impl.
  void write_impl(const char *Ptr, size_t Size) override {
    XXH3_128bits_update(&State, (void *)Ptr, Size);
  }

public:
  raw_xxhash_stream() : raw_ostream() { XXH3_128bits_reset(&State); }

  std::array<uint8_t, 16> hash() {
    flush();

    XXH128_hash_t digest = XXH3_128bits_digest(&State);
    std::array<uint8_t, 16> result;
    memcpy(&result[0], (void *)&digest.low64, 8);
    memcpy(&result[8], (void *)&digest.high64, 8);
    return result;
  }

  uint64_t current_pos() const override { return 0; }
};

FailureOr<std::string> getBytecodeHash(mlir::Operation *op) {
  auto unknownLoc = UnknownLoc::get(op->getContext());

  auto *builtin = op->getContext()->getLoadedDialect<mlir::BuiltinDialect>();
  BytecodeDialectInterface *iface =
      builtin->getRegisteredInterface<BytecodeDialectInterface>();

  BytecodeWriterConfig config;
  config.attachAttributeCallback(
      [&](Attribute entryValue, std::optional<StringRef> &dialectGroupName,
          DialectBytecodeWriter &writer) -> LogicalResult {
        // Map all locations attributes to UnknownLoc.
        if (isa<LocationAttr>(entryValue))
          entryValue = unknownLoc;
        return iface->writeAttribute(entryValue, writer);
      });

  raw_xxhash_stream ostream;
  if (failed(mlir::writeBytecodeToFile(op, ostream, config)))
    return op->emitError("Failed to write bytecode");

  SmallString<32> output;
  llvm::toHex(ostream.hash(), /*LowerCase=*/true, output);
  return std::string(output);
}

} // namespace M
