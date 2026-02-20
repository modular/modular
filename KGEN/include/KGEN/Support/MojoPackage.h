//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file provides utilities for reading/writing Mojo package files.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_SUPPORT_MOJOPACKAGE_H
#define KGEN_SUPPORT_MOJOPACKAGE_H

#include "Config/Version.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "llvm/Support/MemoryBufferRef.h"

namespace M::KGEN {

struct MojoPackageVersion final {
  int major = 0;
  int minor = 0;
  int patch = 0;
  std::string label = "";

  MojoPackageVersion() = default;
  MojoPackageVersion(int maj, int min, int pat)
      : major(maj), minor(min), patch(pat) {}

  MojoPackageVersion(const M::ModularVersion &version)
      : major(version.major), minor(version.minor), patch(version.patch),
        label(version.label) {}

  bool operator<(const MojoPackageVersion &other) const {
    return std::tie(major, minor, patch) <
           std::tie(other.major, other.minor, other.patch);
  }

  bool operator>(const MojoPackageVersion &other) const {
    return std::tie(major, minor, patch) >
           std::tie(other.major, other.minor, other.patch);
  }

  bool operator==(const MojoPackageVersion &other) const {
    return std::tie(major, minor, patch) ==
           std::tie(other.major, other.minor, other.patch);
  }

  // Format version for display
  std::string toString() const {
    std::string result = std::to_string(major) + "." + std::to_string(minor) +
                         "." + std::to_string(patch);
    if (!label.empty())
      result += label;
    return result;
  }
};

/// Represents the header section of a Mojo package file, coming before the MLIR
/// section.
struct MojoPackageHeader {
  MojoPackageVersion mojoVersion;
  MojoPackageVersion modularVersion;
  std::string mlirChecksum;
  int version = 1;
  size_t headerSize;

  size_t getSizeInBytes() const { return headerSize; }
  void dump() const;
};

/// Write the bytecode for the given operation to the provided output stream as
/// a Mojo package file. For streams where it matters, the given stream should
/// be in "binary" mode.
LogicalResult writeBinaryPackage(Operation *op, raw_ostream &os);

/// Write the bytecode for the given operation to the provided output stream as
/// a Mojo package file. For streams where it matters, the given stream should
/// be in "binary" mode.
/// Note: public visibility, intended only for round-trip unit testing
LogicalResult writeBinaryPackage(Operation *op, MojoPackageVersion &mojoVer,
                                 MojoPackageVersion &modularVer,
                                 StringRef mlirChecksum, raw_ostream &os);

/// Returns whether the memory buffer points to a valid Mojo package
/// (.mojopkg/.📦) file. Checks only the magic bytes at the beginning of the
/// buffer.
bool isMojoPackage(llvm::MemoryBufferRef buffer);

/// Returns whether the Mojo package (represented by its header) is compatible
/// with the current compiler.
bool isCompatiblePackage(const MojoPackageHeader &header);

/// Returns whether the Mojo package (represented by its header) is compatible
/// with the current compiler, and returns a message explaining any cause of
/// incompatibility.
ErrorOrSuccess checkCompatiblePackage(const MojoPackageHeader &header);

/// Compares two (package) versions and returns how 'other' compares to the
/// 'base' with a human-readable message on inequality. Optionally takes
/// human-readable names for each version and will add those to the message.
ErrorOrSuccess checkVersion(const MojoPackageVersion &base,
                            const MojoPackageVersion &other,
                            llvm::StringRef baseName = "",
                            llvm::StringRef otherName = "");

/// Reads and returns the Mojo package header section of a Mojo package file.
/// Returns an Error on failure. The buffer is read-only; the pointer to
/// the MLIR section can be computed by offsetting the buffer by the size of the
/// returned header (MojoPackageHeader::getSizeInBytes).
ErrorOr<MojoPackageHeader>
readBinaryPackageHeader(llvm::MemoryBufferRef buffer);

// Read a Mojo package, returning both the header and a buffer reference
// pointing to the MLIR section.
ErrorOr<std::pair<MojoPackageHeader, llvm::MemoryBufferRef>>
getMLIRBufferAndHeaderFromPackage(llvm::MemoryBufferRef buffer);

// Read a Mojo package, returning the buffer reference pointing to the MLIR
// section if the header is compatible, or else an error if
// ignoreIncompatiblePackages is false.
ErrorOr<llvm::MemoryBufferRef>
getMLIRBufferFromPackage(llvm::MemoryBufferRef buffer,
                         bool ignoreIncompatiblePackages);

} // namespace M::KGEN

#endif // KGEN_SUPPORT_MOJOPACKAGE_H
