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
  SmallVector<char, 0> outFilePathVec;
  std::error_code err =
      llvm::sys::fs::createUniqueFile(model, fd, outFilePathVec);
  if (err)
    return Error(err.message());

  return TempFile{fd,
                  std::string{outFilePathVec.data(), outFilePathVec.size()}};
}

TempFile::TempFile(TempFile &&other)
    : fd(other.fd), path(other.path), keepFile(other.keepFile) {
  other.keepFile = true;
}

TempFile::~TempFile() {
  if (!keepFile) {
    llvm::sys::fs::closeFile(fd);
    std::filesystem::remove(path);
  }
}

size_t TempFile::getSize() { return std::filesystem::file_size(path); }
