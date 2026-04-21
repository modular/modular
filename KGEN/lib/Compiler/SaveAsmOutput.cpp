//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/Compiler/SaveAsmOutput.h"
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
// Offload output naming helpers
//===----------------------------------------------------------------------===//

llvm::StringRef offloadAsmExt(const llvm::Triple &triple) {
  return triple.isNVPTX()        ? ".ptx"
         : triple.isAMDGCN()     ? ".amdgcn"
         : isMetalTriple(triple) ? ".ll"
                                 : ".s";
}

llvm::StringRef offloadLLVMExt(const llvm::Triple &triple) {
  return triple.isNVPTX()        ? ".nvptx.ll"
         : triple.isAMDGCN()     ? ".amdgcn.ll"
         : isMetalTriple(triple) ? ".metal.ll"
                                 : ".ll";
}

std::string reserveOffloadOutputBaseName(mlir::StringAttr rawName,
                                         llvm::StringRef ext,
                                         llvm::StringMap<int> &nameCountMap) {
  llvm::StringRef name = rawName.getValue();
  assert(
      !name.empty() &&
      llvm::all_of(name, [](char c) { return llvm::isAlnum(c) || c == '_'; }) &&
      "Output symbol name must be non-empty and contain only alnum/underscore");
  std::string nameKey = (name + ext).str();
  int &count = nameCountMap[nameKey];
  std::string baseName = name.str();
  if (count != 0)
    baseName += "_" + std::to_string(count);
  ++count;
  return baseName;
}

std::string offloadOutputPath(llvm::StringRef prefix, llvm::StringRef baseName,
                              llvm::StringRef ext) {
  return (prefix + "_" + baseName + ext).str();
}

//===----------------------------------------------------------------------===//
// Pending offload writes
//===----------------------------------------------------------------------===//

ErrorOrSuccess flushOffloadWrites(mlir::ModuleOp module) {
  auto attr =
      module->getAttrOfType<mlir::DictionaryAttr>(kOffloadWritesAttrName);
  if (!attr)
    return success();
  for (auto entry : attr) {
    llvm::StringRef path = entry.getName().strref();
    llvm::StringRef content =
        mlir::cast<mlir::StringAttr>(entry.getValue()).strref();
    auto outFile = mlir::openOutputFile(path);
    if (!outFile)
      return Error("could not open offload output file '" + path + "'");
    outFile->os() << content;
    outFile->keep();
  }
  module->removeAttr(kOffloadWritesAttrName);
  return success();
}

} // namespace M::KGEN
