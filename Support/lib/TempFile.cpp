//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/TempFile.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FileSystem.h"
#include <filesystem>

using namespace M;

ErrorOr<TempFile> TempFile::create(StringRef model) {
  int fd;
  SmallString<0> outFilePathVec;
  std::error_code err =
      llvm::sys::fs::createUniqueFile(model, fd, outFilePathVec);
  if (err)
    return Error(err.message());

  return TempFile{fd, outFilePathVec.str().str()};
}

TempFile::TempFile(TempFile &&other)
    : fd(other.fd), path(std::move(other.path)), keepFile(other.keepFile) {
  other.fd = -1;
}

TempFile::~TempFile() {
  if (fd != -1)
    llvm::sys::fs::closeFile((llvm::sys::fs::file_t &)fd);

  if (!keepFile)
    std::filesystem::remove(path);
}

ErrorOr<size_t> TempFile::getSize() {
  std::error_code ec;
  uintmax_t size = std::filesystem::file_size(path, ec);
  if (size == (uintmax_t)-1)
    return Error(ec.message());

  return size;
}
