//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_COMPILER_SAVEASMOUTPUT_H
#define KGEN_COMPILER_SAVEASMOUTPUT_H

/// Helpers for writing GPU and host ASM output files.
///
/// Includes two families:
///   1. Save-temps helpers (writeBytesToTempWithHash, writeTempModule) —
///      originally in ObjectCompiler.cpp, shared here so both ObjectCompiler
///      and KGENCompiler can use them.
///   2. GPU ASM output naming and cache-boundary staging helpers
///      (gpuAsmExt, reserveGpuAsmBaseName, gpuAsmOutputPath,
///      kGpuAsmWritesAttrName, flushGpuAsmWrites) — originally in
///      KGENCompiler.cpp, moved here to keep KGENCompiler.cpp focused on
///      pipeline orchestration.

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "Support/ErrorOr.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
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
// GPU ASM output naming helpers
//===----------------------------------------------------------------------===//

/// Return the file extension for GPU ASM output for \p triple.
/// Metal (air64) uses LLVM IR text as its human-readable "assembly".
llvm::StringRef gpuAsmExt(const llvm::Triple &triple);

/// Reserve a disambiguated base name for a GPU ASM output file and return it.
/// The base name is the sanitized kernel source name with an optional _N suffix
/// to resolve collisions among same-named kernel instantiations.  The extension
/// is included in the collision key so PTX and Metal files for the same kernel
/// name count independently.  \p nameCountMap is updated in place.
std::string reserveGpuAsmBaseName(mlir::StringAttr rawName,
                                  TargetInfoAttr target,
                                  llvm::StringMap<int> &nameCountMap);

/// Return the output path for a GPU ASM file.
/// Format: <prefix>_<baseName><ext>, where ext is determined by \p target.
std::string gpuAsmOutputPath(llvm::StringRef prefix, TargetInfoAttr target,
                             llvm::StringRef baseName);

//===----------------------------------------------------------------------===//
// GPU ASM cache-boundary staging
//===----------------------------------------------------------------------===//

/// Module attribute that stages GPU ASM file writes across the
/// Cache::cachedTransform boundary.  Set by compileOffloads() inside the pass
/// pipeline; consumed and removed by flushGpuAsmWrites() after cachedTransform
/// returns.
inline constexpr llvm::StringLiteral kGpuAsmWritesAttrName =
    "kgen.gpu_asm_writes";

/// Write all GPU ASM files staged on \p module by compileOffloads() and remove
/// the staging attribute.  Must be called after cachedTransform on both the
/// hit path (deserialized module) and the miss path (freshly compiled module)
/// so the files are produced regardless of cache outcome.
ErrorOrSuccess flushGpuAsmWrites(mlir::ModuleOp module);

} // namespace M::KGEN

#endif // KGEN_COMPILER_SAVEASMOUTPUT_H
