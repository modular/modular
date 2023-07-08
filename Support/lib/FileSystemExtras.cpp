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
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"

using namespace M;

ErrorOr<std::filesystem::path>
M::writeFileAtomically(const std::filesystem::path &filePath,
                       llvm::function_ref<void(raw_ostream &)> writeContent) {
  std::string filePathStr = filePath.string();

  // A helper function to write the content into the file.
  auto writeFile = [&]() -> ErrorOr<std::filesystem::path> {
    llvm::Error err = llvm::writeToOutput(filePathStr, [&](raw_ostream &os) {
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

// llvm::sys::Process has a function called `llvm::sys::Process::FindInEnvPath`
// which looks for files (and files only) in PATH like environment variables.
// The version here is inspired by the original and has a similar contract but
// looks only for directories instead.
std::optional<std::string>
M::findDirInEnvPath(StringRef subdirName, StringRef envName, char separator) {
  assert(!llvm::sys::path::is_absolute(subdirName));
  std::optional<std::string> optPath = llvm::sys::Process::GetEnv(envName);
  if (!optPath)
    return {};

  const char envPathSeparatorStr[] = {separator, '\0'};
  SmallVector<StringRef, 8> dirs;
  StringRef(*optPath).split(dirs, envPathSeparatorStr);

  for (StringRef dir : dirs) {
    if (dir.empty())
      continue;

    SmallString<128> dirPath(dir);
    llvm::sys::path::append(dirPath, subdirName);
    if (llvm::sys::fs::exists(Twine(dirPath)) &&
        llvm::sys::fs::is_directory(Twine(dirPath))) {
      return std::string(dirPath);
    }
  }

  return std::nullopt;
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
