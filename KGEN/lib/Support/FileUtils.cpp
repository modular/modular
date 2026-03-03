//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/Support/FileUtils.h"

#include "mlir/Support/FileUtilities.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/xxhash.h"

using namespace M;
using namespace KGEN;

mlir::LogicalResult
M::KGEN::writeBytesToTempWithHash(const std::string &saveTempsPrefix,
                                  const std::string &postfix,
                                  llvm::StringRef buf) {
  if (saveTempsPrefix.empty())
    return mlir::success();

  // Include unique hash as part of name.
  assert(sizeof(uint8_t) == sizeof(char) && "Assume char is 8 bits");
  llvm::XXH128_hash_t hash =
      llvm::xxh3_128bits(llvm::arrayRefFromStringRef(buf));
  std::string outPath =
      saveTempsPrefix + "." + llvm::utohexstr(hash.high64, /*LowerCase=*/true) +
      llvm::utohexstr(hash.low64, /*LowerCase=*/true) + postfix;

  auto outFile = mlir::openOutputFile(outPath);
  if (!outFile)
    return mlir::failure();
  outFile->os() << buf;
  outFile->keep();
  return mlir::success();
}

mlir::LogicalResult
M::KGEN::writeBytesToTempWithHash(const std::string &saveTempsPrefix,
                                  const std::string &postfix,
                                  llvm::MemoryBufferRef buf) {
  return writeBytesToTempWithHash(saveTempsPrefix, postfix, buf.getBuffer());
}
