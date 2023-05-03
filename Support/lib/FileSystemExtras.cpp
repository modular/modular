//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/FileSystemExtras.h"
#include "Support/ErrorOr.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/LockFileManager.h"

using namespace M;

ErrorOr<std::filesystem::path>
M::writeFileAtomically(const std::filesystem::path &filePath,
                       llvm::function_ref<void(raw_ostream &)> writeContent) {
  std::string filePathStr = filePath.string();

  // A helper function to write the content into the file.
  auto writeFile = [&]() -> ErrorOr<std::filesystem::path> {
    llvm::Error err = llvm::writeFileAtomically(
        filePathStr + "-%%%%%%%%", filePathStr, [&](raw_ostream &os) {
          writeContent(os);
          return llvm::Error::success();
        });
    if (err)
      return Error(llvm::toString(std::move(err)));
    return filePath;
  };

  // Lock or wait for the file to be able to write to it.
  while (true) {
    llvm::LockFileManager lockManager(filePathStr);
    switch (lockManager) {
    case llvm::LockFileManager::LFS_Error:
      return Error("unable to take lock file for '" + filePathStr +
                   "': " + lockManager.getErrorMessage());
    case llvm::LockFileManager::LFS_Owned:
      // We got the lock, and can build the file.
      return writeFile();

    case llvm::LockFileManager::LFS_Shared:
      // Another process is touching the file, handle the different
      // outcomes of this below.
      break;
    }

    // Wait for the other process to finish touching the file.
    switch (lockManager.waitForUnlock()) {
    case llvm::LockFileManager::Res_Success:
      // We now have the lock file, and can proceed to build the file if the
      // other process didn't do it.
      return writeFile();
    case llvm::LockFileManager::Res_OwnerDied:
      // The owner died, try again to take the file.
      continue;
    case llvm::LockFileManager::Res_Timeout:
      // We timed out when trying to acquire the lock for the file.
      // TODO: We could try again, but the default timeout is 1.5 minutes.
      return Error("timed out waiting for lock file for '" + filePathStr + "'");
    }
  }
  return filePath;
}

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
  if (fd != -1) {
    llvm::sys::fs::file_t nativeID = llvm::sys::fs::convertFDToNativeFile(fd);
    llvm::sys::fs::closeFile(nativeID);
  }

  if (!keepFile) {
    std::error_code ec;
    std::filesystem::remove(path, ec);
  }
}

ErrorOr<size_t> TempFile::getSize() {
  std::error_code ec;
  uintmax_t size = std::filesystem::file_size(path, ec);
  if (size == (uintmax_t)-1)
    return Error(ec.message());

  return size;
}
