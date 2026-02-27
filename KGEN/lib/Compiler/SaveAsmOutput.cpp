//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/Compiler/SaveAsmOutput.h"
#include "KGEN/Support/NameMangling.h"
#include "KGEN/ToolCommon/CompilationOptions.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/xxhash.h"
#include <cassert>

namespace M::KGEN {

//===----------------------------------------------------------------------===//
// Save-temps helpers
//===----------------------------------------------------------------------===//

mlir::LogicalResult writeBytesToTempWithHash(const std::string &saveTempsPrefix,
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

mlir::LogicalResult writeBytesToTempWithHash(const std::string &saveTempsPrefix,
                                             const std::string &postfix,
                                             llvm::MemoryBufferRef buf) {
  return writeBytesToTempWithHash(saveTempsPrefix, postfix, buf.getBuffer());
}

//===----------------------------------------------------------------------===//
// GPU ASM output naming helpers
//===----------------------------------------------------------------------===//

llvm::StringRef gpuAsmExt(const llvm::Triple &triple) {
  return triple.isNVPTX()        ? ".ptx"
         : triple.isAMDGCN()     ? ".amdgcn"
         : isMetalTriple(triple) ? ".ll"
                                 : ".s";
}

std::string reserveGpuAsmBaseName(mlir::StringAttr rawName,
                                  TargetInfoAttr target,
                                  llvm::StringMap<int> &nameCountMap) {
  std::string sanitized = sanitizeSymbolToAlnum(rawName).getValue().str();
  llvm::StringRef ext = gpuAsmExt(llvm::Triple(target.getTripleStr()));
  std::string nameKey = sanitized + ext.str();
  int &count = nameCountMap[nameKey];
  std::string baseName =
      count == 0 ? sanitized : sanitized + "_" + std::to_string(count);
  ++count;
  return baseName;
}

std::string gpuAsmOutputPath(llvm::StringRef prefix, TargetInfoAttr target,
                             llvm::StringRef baseName) {
  return (prefix + "_" + baseName +
          gpuAsmExt(llvm::Triple(target.getTripleStr())))
      .str();
}

//===----------------------------------------------------------------------===//
// GPU ASM cache-boundary staging
//===----------------------------------------------------------------------===//

ErrorOrSuccess flushGpuAsmWrites(mlir::ModuleOp module) {
  auto attr =
      module->getAttrOfType<mlir::DictionaryAttr>(kGpuAsmWritesAttrName);
  if (!attr)
    return success();
  for (auto entry : attr) {
    llvm::StringRef path = entry.getName().strref();
    llvm::StringRef content =
        mlir::cast<mlir::StringAttr>(entry.getValue()).strref();
    auto outFile = mlir::openOutputFile(path);
    if (!outFile)
      return Error("could not open GPU ASM output file '" + path + "'");
    outFile->os() << content;
    outFile->keep();
  }
  module->removeAttr(kGpuAsmWritesAttrName);
  return success();
}

} // namespace M::KGEN
