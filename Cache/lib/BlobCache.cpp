//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Cache/BlobCache.h"
#include "Config/Config.h"
#include "Support/HMAC.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/LockFileManager.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"

using namespace M;

//===----------------------------------------------------------------------===//
// Hashing
//===----------------------------------------------------------------------===//

std::string M::Detail::finalizeBlobKeyHash(StringRef hash) {
  // Incorporate the current version into the hash.
  return std::to_string(
      size_t(llvm::hash_combine(hash, StringRef(MODULAR_VERSION_STRING))));
}

//===----------------------------------------------------------------------===//
// BlobCacheBackend
//===----------------------------------------------------------------------===//

ErrorOrSuccess BlobCacheBackend::insert(StringRef keyHash,
                                        llvm::MemoryBufferRef obj) {
  RETURN_ERROR(insertImpl(keyHash, obj));

  if (delegate)
    RETURN_ERROR(delegate->insert(keyHash, obj));

  return success();
}

bool BlobCacheBackend::contains(StringRef keyHash) {
  if (containsImpl(keyHash))
    return true;
  if (delegate)
    return delegate->contains(keyHash);
  return false;
}

CacheFindResult BlobCacheBackend::find(StringRef keyHash) {
  if (containsImpl(keyHash))
    return this->findImpl(keyHash);

  if (!delegate)
    return CacheFindResult::error("could not find item '" + keyHash + "'");

  auto itemOr = delegate->find(keyHash);
  if (itemOr.isError())
    return CacheFindResult::error(itemOr.takeError());

  std::unique_ptr<llvm::MemoryBuffer> item = itemOr.takeValue();
  // Store the item in our cache level so we can get a cache hit later.
  if (auto err = insertImpl(keyHash, *item))
    return CacheFindResult::error(err.takeError());

  // Return the item.
  return CacheFindResult::value(std::move(item));
}

//===----------------------------------------------------------------------===//
// InMemoryBackend
//===----------------------------------------------------------------------===//

namespace {
/// Provides an in-memory backend that stores memory buffers in an
/// llvm::StringMap.
struct InMemoryBackend : public BlobCacheBackend {
  ErrorOrSuccess insertImpl(StringRef keyHash,
                            llvm::MemoryBufferRef obj) override {
    // Store the item in this cache.
    cache[keyHash] = llvm::MemoryBuffer::getMemBufferCopy(
        obj.getBuffer(), obj.getBufferIdentifier());
    return success();
  }

  bool containsImpl(StringRef keyHash) const override {
    return cache.count(keyHash);
  }

  CacheFindResult findImpl(StringRef keyHash) const override {
    auto found = cache.find(keyHash);
    if (found == cache.end())
      return CacheFindResult::error("could not find item '" + keyHash + "'");

    // Create a memory buffer that aliases this same data.
    auto &mbuf = (*found).second;
    return CacheFindResult::value(llvm::MemoryBuffer::getMemBuffer(
        mbuf->getBuffer(), mbuf->getBufferIdentifier()));
  }

  llvm::StringMap<std::unique_ptr<llvm::MemoryBuffer>> cache;
};
} // namespace

std::unique_ptr<BlobCacheBackend> M::getInMemoryBackend() {
  return std::make_unique<InMemoryBackend>();
}

//===----------------------------------------------------------------------===//
// FilesystemBackend
//===----------------------------------------------------------------------===//

namespace {
/// Provides a filesystem-backed backend that stores the buffers in binary files
/// on disk.
struct FilesystemBackend : public BlobCacheBackend {
  explicit FilesystemBackend(const std::filesystem::path &basePath)
      : basePath(basePath.string()) {}

  ErrorOrSuccess insertImpl(StringRef keyHash,
                            llvm::MemoryBufferRef obj) override {
    // Get the absolute path and create any directories we need to create.
    std::filesystem::path filePath = getAbsolutePathForKey(keyHash);
    std::error_code err;
    std::filesystem::create_directories(filePath.parent_path(), err);
    if (err)
      return Error(err.message());
    std::string filePathStr = filePath.string();

    // Functor used when we actually need to write out the file.
    auto writeFile = [&]() -> ErrorOrSuccess {
      llvm::Error err = llvm::writeFileAtomically(
          filePathStr + "-%%%%%%%%", filePathStr, [&](raw_ostream &os) {
            // Copy the data into the file buffer.
            os.write(obj.getBufferStart(), obj.getBufferSize());

            // Compute and copy the HMAC as well.
            SHA256Hash hash = hmacSHA256(obj.getBuffer(), kIntegrityKey);
            os.write((const char *)hash.data(), hash.size());
            return llvm::Error::success();
          });
      return err ? Error(llvm::toString(std::move(err))) : ErrorOrSuccess();
    };

    // Safely process creating the file, taking into account that we may have
    // different processes trying to produce this file in parallel.
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
        if (containsImpl(keyHash))
          return success();
        return writeFile();
      case llvm::LockFileManager::Res_OwnerDied:
        // The owner died, try again to take the file.
        continue;
      case llvm::LockFileManager::Res_Timeout:
        // We timed out when trying to acquire the lock for the file.
        // TODO: We could try again, but the default timeout is 1.5 minutes.
        return Error("timed out waiting for lock file for '" + filePathStr +
                     "'");
      }
    }
  }

  bool containsImpl(StringRef keyHash) const override {
    auto abs = getAbsolutePathForKey(keyHash);
    return std::filesystem::exists(abs) && !std::filesystem::is_directory(abs);
  }

  CacheFindResult findImpl(StringRef keyHash) const override {
    // Get the file path and open it.
    std::filesystem::path filePath = getAbsolutePathForKey(keyHash);
    std::string filePathStr = filePath.string();
    auto fileOr = llvm::MemoryBuffer::getFile(filePathStr);

    // If the file doesn't exist, or it's empty, return an error.
    if (!fileOr)
      return CacheFindResult::error(Error(fileOr.getError().message()));
    if ((*fileOr)->getBufferSize() == 0)
      return CacheFindResult::error("file '" + Twine(filePath.string()) +
                                    "' exists, but is empty");

    std::unique_ptr<llvm::MemoryBuffer> fileBuf = std::move(*fileOr);
    StringRef contentsAndHMAC = fileBuf->getBuffer();

    // Get a StringRef of the contents without the HMAC.
    StringRef contents = contentsAndHMAC.drop_back(sha256Bytes);
    SHA256Hash computedHMAC = hmacSHA256(contents, kIntegrityKey);
    StringRef storedHMAC = contentsAndHMAC.take_back(sha256Bytes);

    // Check the computed hmac against the one in the file.
    if (memcmp(computedHMAC.data(), storedHMAC.data(), sha256Bytes)) {
      return CacheFindResult::error(
          "corrupted file: stored hash and computed hash did not "
          "match for file '" +
          Twine(filePath.string()) + "'");
    }

    // Now that we've verified the integrity of the file, return a memory buffer
    // that holds just the contents.
    fileOr = llvm::MemoryBuffer::getFileSlice(filePathStr, contents.size(),
                                              /*Offset=*/0);
    if (!fileOr)
      return CacheFindResult::error(Error(fileOr.getError().message()));
    return CacheFindResult::value(std::move(*fileOr));
  }

  std::filesystem::path getAbsolutePathForKey(StringRef keyHash) const {
    std::filesystem::path filepath(basePath);
    filepath /= keyHash.str();

    return std::filesystem::absolute(filepath);
  }

  /// This is a CSPRNG-generated 32-byte string. It's used for integrity
  /// checking in the HMAC.
  static constexpr llvm::StringLiteral kIntegrityKey =
      "bedcaea9f09fa9fe565a8088ea66547c06c7c8e9c47fa46e0fb768a157d640a6";
  /// The base path for the filesystem cache.
  std::string basePath;
};
} // namespace

std::unique_ptr<BlobCacheBackend>
M::getFilesystemBackend(const std::filesystem::path &basePath) {
  return std::make_unique<FilesystemBackend>(basePath);
}

std::unique_ptr<BlobCacheBackend>
M::getDefaultBackendChain(const std::filesystem::path &basePath) {
  auto backend = getInMemoryBackend();

  /* TODO: Disabled for now while we debug the filesystem backend
           implementation (c.f. issue #4394)
  // Default to be in the `.derived` folder if we can.
  std::error_code ec;
  std::filesystem::path derived = std::filesystem::absolute(
      llvm::sys::Process::GetEnv("MODULAR_DERIVED_PATH").value_or("."), ec);
  if (ec) {
    llvm::errs() << ec.message();
    return nullptr;
  }

  std::filesystem::path base = basePath;
  if (!base.is_absolute())
    base = derived / basePath;

  backend->setDelegate(getFilesystemBackend(base));
  */
  return backend;
}
