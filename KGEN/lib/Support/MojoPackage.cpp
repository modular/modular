//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file provides utilities for reading/writing Mojo package files.
//
//===----------------------------------------------------------------------===//

#include "KGEN/Support/MojoPackage.h"
#include "KGEN/DialectChecksum/DialectChecksum.h"
#include "Support/Error.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/NativeFormatting.h"

using namespace M;
using namespace KGEN;

/// Returns whether the Mojo package (represented by its header) is compatible
/// with the current compiler.
ErrorOrSuccess
M::KGEN::checkCompatiblePackage(const MojoPackageHeader &header) {
  // TODO: Enable compatibility checking
  return success();
}

ErrorOrSuccess M::KGEN::checkVersion(const MojoPackageVersion &base,
                                     const MojoPackageVersion &other,
                                     llvm::StringRef baseName,
                                     llvm::StringRef otherName) {
  // Note: these comparisons ignore labels
  if (base == other)
    return success();
  StringRef what = base > other ? "newer" : "older";
  return Error(baseName + (baseName.empty() ? "" : " ") + base.toString() +
               " is " + what + " than " + otherName +
               (otherName.empty() ? "" : " ") + other.toString());
}

bool M::KGEN::isCompatiblePackage(const MojoPackageHeader &header) {
  return !checkCompatiblePackage(header).isError();
}

template <typename T>
static void writeInt(llvm::raw_ostream &os, uint64_t x) {
  SmallVector<char> buffer(sizeof(T));
  llvm::support::endian::write<T>(buffer.data(), static_cast<T>(x),
                                  llvm::endianness::little);
  os << buffer;
}

static void writeVersion(MojoPackageVersion version, llvm::raw_ostream &os) {
  writeInt<uint8_t>(os, version.major);
  writeInt<uint8_t>(os, version.minor);
  writeInt<uint16_t>(os, version.patch);
  // Write out the label, null terminated.
  os << version.label << '\0';
}

LogicalResult M::KGEN::writeBinaryPackage(Operation *op,
                                          MojoPackageVersion &mojoVer,
                                          MojoPackageVersion &modularVer,
                                          StringRef mlirChecksum,
                                          llvm::raw_ostream &os) {
  [[maybe_unused]] auto streamPos = os.tell();
  // Write out the MojoPackage header
  os << "MPKG";
  // MojoPackage format, version 1
  writeInt<uint8_t>(os, 1);
  os << "..."; // plus 3 reserved bytes.

  // Write the Mojo version (not currently set; ignored)
  writeVersion(mojoVer, os);

  // Write the Modular version
  writeVersion(modularVer, os);

  // Write the nul-terminated MLIR checksum
  os << mlirChecksum << '\0';

  // Align the header size to 8 bytes.
  auto bytesWritten = os.tell() - streamPos;
  auto paddingBytes = llvm::alignTo(bytesWritten, 8) - bytesWritten;
  while (paddingBytes--)
    writeInt<uint8_t>(os, 0);

  // Now serialize the MLIR.
  return mlir::writeBytecodeToFile(op, os);
}

LogicalResult M::KGEN::writeBinaryPackage(Operation *op,
                                          llvm::raw_ostream &os) {
  // The Mojo version - not currently set; ignored
  MojoPackageVersion mojoVersion{0, 0, 0};
  // The Modular version - currently only for public release builds
  MojoPackageVersion modularVersion = M::getModularVersion();
  // The MLIR checksum
  StringRef mlirChecksum = M::getMojoMlirDialectChecksum();
  return writeBinaryPackage(op, mojoVersion, modularVersion, mlirChecksum, os);
}

bool M::KGEN::isMojoPackage(llvm::MemoryBufferRef buffer) {
  return buffer.getBuffer().starts_with("MPKG");
}

template <typename T>
ErrorOr<std::pair<T, llvm::StringRef>> readInt(llvm::StringRef buffer) {
  if (buffer.size() < sizeof(T))
    return Error("read past end of buffer");
  return std::make_pair<T, llvm::StringRef>(
      llvm::support::endian::read<T, llvm::support::unaligned>(
          buffer.data(), llvm::endianness::little),
      buffer.drop_front(sizeof(T)));
}

static ErrorOr<std::pair<MojoPackageVersion, llvm::StringRef>>
readVersion(llvm::StringRef buffer) {
  MojoPackageVersion version;

  // Major
  if (auto err = readInt<uint8_t>(buffer))
    return err.takeError();
  else
    std::tie(version.major, buffer) = *err;
  // Minor
  if (auto err = readInt<uint8_t>(buffer))
    return err.takeError();
  else
    std::tie(version.minor, buffer) = *err;
  // Patch
  if (auto err = readInt<uint16_t>(buffer))
    return err.takeError();
  else
    std::tie(version.patch, buffer) = *err;

  // Parse the NUL terminated string matching the version label.
  version.label = buffer.take_until([](char c) { return c == '\0'; });

  // Check we've reached a nul terminator
  auto remaining = buffer.drop_front(version.label.size());
  if (remaining.empty() || remaining.front() != '\0')
    return Error("invalid version encoding");

  return std::make_pair(version, remaining.drop_front());
}

ErrorOr<MojoPackageHeader>
M::KGEN::readBinaryPackageHeader(llvm::MemoryBufferRef buffer) {
  llvm::StringRef bufferStr = buffer.getBuffer();
  if (!isMojoPackage(buffer))
    return Error("invalid magic bytes");

  // A package header must be at least 8 bytes, to begin with. We'll keep
  // checking as we go.
  if (bufferStr.size() < 8)
    return Error("invalid header size");

  MojoPackageHeader header;
  // Skip past the 4 magic bytes
  bufferStr = bufferStr.drop_front(4);

  // Read the single-byte encoding version information
  if (auto err = readInt<uint8_t>(bufferStr))
    return err.takeError();
  else
    std::tie(header.version, bufferStr) = *err;

  // Skip past the 3 currently unused bytes.
  bufferStr = bufferStr.drop_front(3);

  auto mojoVersionOrErr = readVersion(bufferStr);
  if (mojoVersionOrErr.isError())
    return mojoVersionOrErr.takeError();
  std::tie(header.mojoVersion, bufferStr) = *mojoVersionOrErr;

  auto modularVersionOrErr = readVersion(bufferStr);
  if (modularVersionOrErr.isError())
    return modularVersionOrErr.takeError();
  std::tie(header.modularVersion, bufferStr) = *modularVersionOrErr;

  header.mlirChecksum = bufferStr.take_until([](char c) { return c == '\0'; });
  bufferStr = bufferStr.drop_front(header.mlirChecksum.size());

  // Skip past the NUL terminator (as readVersion does for version labels).
  if (bufferStr.empty() || bufferStr.front() != '\0')
    return Error("invalid checksum encoding");
  bufferStr = bufferStr.drop_front(1);

  header.headerSize =
      llvm::alignTo(buffer.getBufferSize() - bufferStr.size(), 8);
  if (buffer.getBufferSize() < header.headerSize)
    return Error("invalid header size");

  return header;
}

// Return a buffer reference from a Mojo package, skipping the header bytes
ErrorOr<std::pair<MojoPackageHeader, llvm::MemoryBufferRef>>
M::KGEN::getMLIRBufferAndHeaderFromPackage(llvm::MemoryBufferRef buffer) {
  auto header = readBinaryPackageHeader(buffer);
  if (header.isError())
    return Error("invalid Mojo package '" + buffer.getBufferIdentifier() +
                 "': " + header.getError());
  // Return the header and a buffer pointing to the start of the MLIR section.
  return std::make_pair(
      *header, llvm::MemoryBufferRef(
                   buffer.getBuffer().drop_front(header->getSizeInBytes()),
                   buffer.getBufferIdentifier()));
}

ErrorOr<llvm::MemoryBufferRef>
M::KGEN::getMLIRBufferFromPackage(llvm::MemoryBufferRef buffer,
                                  bool ignoreIncompatiblePackageErrs) {
  auto mlirBufferAndHeaderOrErr = getMLIRBufferAndHeaderFromPackage(buffer);
  if (mlirBufferAndHeaderOrErr.isError())
    return mlirBufferAndHeaderOrErr.takeError();
  auto &[header, mlirBuffer] = *mlirBufferAndHeaderOrErr;
  if (!ignoreIncompatiblePackageErrs)
    if (auto err = KGEN::checkCompatiblePackage(header))
      return Error("invalid Mojo package '" + buffer.getBufferIdentifier() +
                   "': " + err.takeError().get());
  return mlirBuffer;
}

void MojoPackageHeader::dump() const {
  llvm::dbgs() << "Encoding ver " << version << "\n";
  llvm::dbgs() << "Mojo Version: " << mojoVersion.major << "."
               << mojoVersion.minor << "." << mojoVersion.patch
               << mojoVersion.label << "\n";
  llvm::dbgs() << "Modular Version: " << modularVersion.major << "."
               << modularVersion.minor << "." << modularVersion.patch
               << modularVersion.label << "\n";
  llvm::dbgs() << "MLIR Checksum: "
               << (mlirChecksum.empty() ? "<none>" : mlirChecksum) << "\n";
}
