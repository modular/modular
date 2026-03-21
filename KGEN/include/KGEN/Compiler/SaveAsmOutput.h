//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_COMPILER_SAVEASMOUTPUT_H
#define KGEN_COMPILER_SAVEASMOUTPUT_H

/// Helpers for writing GPU and host output files.
///
/// Includes two families:
///   1. Save-temps helpers (writeBytesToTempWithHash, writeTempModule) —
///      originally in ObjectCompiler.cpp, shared here so both ObjectCompiler
///      and KGENCompiler can use them.
///   2. Helpers for writing files from the kernel offload compilation
///      (offloadAsmExt, reserveOffloadOutputBaseName, offloadOutputPath,
///      kOffloadWritesAttrName, flushOffloadWrites).

#include "Support/ErrorOr.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include <string>

namespace M::KGEN {

//===----------------------------------------------------------------------===//
// Save-temps helpers
//===----------------------------------------------------------------------===//

/// Write \p buf to a content-addressed file at
/// <saveTempsPrefix>.<xxh128><postfix>.  No-op when saveTempsPrefix is empty.
mlir::LogicalResult writeBytesToTempWithHash(const std::string &saveTempsPrefix,
                                             const std::string &postfix,
                                             llvm::StringRef buf);

mlir::LogicalResult writeBytesToTempWithHash(const std::string &saveTempsPrefix,
                                             const std::string &postfix,
                                             llvm::MemoryBufferRef buf);

/// Serialize \p module to a string and save it via writeBytesToTempWithHash.
/// The output path is <saveTempsPrefix><phase>.<hash><fileExt>.
template <typename ModuleT>
mlir::LogicalResult writeTempModule(const std::string &saveTempsPrefix,
                                    const std::string &phase, ModuleT &module,
                                    const std::string &fileExt = ".ll") {
  if (saveTempsPrefix.empty())
    return mlir::success();
  const std::string finalSavePrefix = saveTempsPrefix + phase;
  std::string str;
  llvm::raw_string_ostream ss(str);
  ss << module;
  return writeBytesToTempWithHash(finalSavePrefix, fileExt, str);
}

//===----------------------------------------------------------------------===//
// Offload output naming helpers
//===----------------------------------------------------------------------===//

/// Return the file extension for offload ASM output for \p triple.
/// Metal (air64) uses LLVM IR text as its human-readable "assembly".
llvm::StringRef offloadAsmExt(const llvm::Triple &triple);

/// Return the file extension for offload LLVM IR output for \p triple.
/// Each target gets a distinct extension (.nvptx.ll, .amdgcn.ll, .metal.ll)
/// so that kernels from different targets do not collide in the output
/// directory when multiple accelerators are compiled in one pass.
llvm::StringRef offloadLLVMExt(const llvm::Triple &triple);

/// Reserve a disambiguated base name for an offload output file.
/// \p ext is the file extension (e.g. ".ptx", ".amdgcn", ".ll").
/// The collision key is "<sanitized-name><ext>" so different extensions track
/// independently. \p nameCountMap is updated in place.
std::string reserveOffloadOutputBaseName(mlir::StringAttr rawName,
                                         llvm::StringRef ext,
                                         llvm::StringMap<int> &nameCountMap);

/// Return the output path for an offload kernel file.
/// Format: <prefix>_<baseName><ext>
std::string offloadOutputPath(llvm::StringRef prefix, llvm::StringRef baseName,
                              llvm::StringRef ext);

//===----------------------------------------------------------------------===//
// Pending offload writes
//===----------------------------------------------------------------------===//

/// Module attribute that holds pending offload writes past cachedTransform.
/// compileOffloads() encodes the files it wants to save (.asm or .ll)
/// as a dictionary (path → content) on the module;
/// cachedTransform serializes it into the cache alongside the IR.
/// flushOffloadWrites() reads this attribute after cachedTransform returns,
/// writing files on both the cache-hit path (deserialized module)
/// and the cache-miss path.
inline constexpr llvm::StringLiteral kOffloadWritesAttrName =
    "kgen.offload_debug_files";

/// Flush all pending offload writes on \p module and remove the attribute.
/// Must be called after cachedTransform so files are written on both hit and
/// miss paths.
ErrorOrSuccess flushOffloadWrites(mlir::ModuleOp module);

} // namespace M::KGEN

#endif // KGEN_COMPILER_SAVEASMOUTPUT_H
