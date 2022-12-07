//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Cache/BlobCache.h"
#include "Config/Config.h"
#include "LLCL/Runtime/Algorithms.h"
#include "LLCL/Runtime/Runtime.h"
#include "Support/HMAC.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/LockFileManager.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"

using namespace M;
using namespace Cache;
using namespace LLCL;

//===----------------------------------------------------------------------===//
// BlobCacheBackend
//===----------------------------------------------------------------------===//

AsyncValueRef<ErrorOrSuccess> BlobCacheBackend::insert(BufferRef keyHash,
                                                       BufferRef obj) {
  auto result = AsyncValueRef<ErrorOrSuccess>::allocate(runtime);
  addTask(runtime, [thisRef = copyRCRef(this), keyHash = keyHash.copy(),
                    obj = obj.copy(), result = result.copy()]() {
    if (auto err = thisRef->insertImpl(keyHash->getBuffer(), obj.copy()))
      return result.emplace(err.takeError());

    if (thisRef->delegate) {
      auto insert = thisRef->delegate->insert(keyHash.copy(), obj.copy());
      insert.andThen([thisRef = thisRef.copy(), result = result.copy(),
                      insert = insert.copy()] {
        if (failed(*insert))
          return result.emplace(insert->takeError());
        result.emplace(success());
      });
    } else {
      return result.emplace(success());
    }
  });
  return result;
}

AsyncValueRef<bool> BlobCacheBackend::contains(BufferRef keyHash) {
  auto result = AsyncValueRef<bool>::allocate(runtime);
  addTask(runtime, [thisRef = copyRCRef(this), keyHash = keyHash.copy(),
                    result = result.copy()]() {
    if (thisRef->containsImpl(keyHash->getBuffer()))
      return result.emplace(true);

    if (thisRef->delegate) {
      auto contains = thisRef->delegate->contains(keyHash.copy());
      contains.andThen(
          [thisRef = thisRef.copy(), result = result.copy(),
           contains = contains.copy()] { return result.emplace(*contains); });
    } else {
      return result.emplace(false);
    }
  });
  return result;
}

AsyncValueRef<CacheFindResult> BlobCacheBackend::find(BufferRef keyHash) {
  auto result = AsyncValueRef<CacheFindResult>::allocate(runtime);
  addTask(
      runtime,
      [thisRef = copyRCRef(this), keyHash = keyHash.copy(),
       result = result.copy()]() -> void {
        if (thisRef->containsImpl(keyHash->getBuffer()))
          return result.emplace(thisRef->findImpl(keyHash->getBuffer()));

        if (!thisRef->delegate)
          return result.emplace(CacheFindResult::notInCache());

        auto itemOr = thisRef->delegate->find(keyHash.copy());
        itemOr.andThen([thisRef = thisRef.copy(), keyHash = keyHash.copy(),
                        result = result.copy(), itemOr = itemOr.copy()]() {
          if (itemOr->isError())
            return result.emplace(CacheFindResult::error(itemOr->takeError()));

          // Delegate doesn't have it either!
          if (!itemOr->hasValue())
            return result.emplace(CacheFindResult::notInCache());

          BufferRef item = itemOr->takeValue();

          // Store the item in our cache level so we can get a cache hit
          // later.
          if (auto err = thisRef->insertImpl(keyHash->getBuffer(), item.copy()))
            return result.emplace(CacheFindResult::error(err.takeError()));

          // Return the item.
          return result.emplace(CacheFindResult::value(std::move(item)));
        });
      });

  return result;
}

AsyncValueRef<ErrorOrSuccess> BlobCacheBackend::clear() {
  auto result = AsyncValueRef<ErrorOrSuccess>::allocate(runtime);
  addTask(runtime, [thisRef = copyRCRef(this), result = result.copy()]() {
    if (auto err = thisRef->clearImpl())
      return result.emplace(err.takeError());

    if (thisRef->delegate) {
      auto clear = thisRef->delegate->clear();
      clear.andThen([thisRef = thisRef.copy(), result = result.copy(),
                     clear = clear.copy()] {
        if (failed(*clear))
          return result.emplace(clear->takeError());
        result.emplace(success());
      });
    } else {
      return result.emplace(success());
    }
  });
  return result;
}

//===----------------------------------------------------------------------===//
// InMemoryBackend
//===----------------------------------------------------------------------===//

namespace {
/// Provides an in-memory backend that stores memory buffers in an
/// llvm::StringMap.
struct InMemoryBackend : public BlobCacheBackend {
  InMemoryBackend(LLCL::Runtime &runtime) : BlobCacheBackend(runtime) {}

  ErrorOrSuccess insertImpl(StringRef keyHash, BufferRef obj) override {
    // Store the item in this cache.
    cache[keyHash] = std::move(obj);
    return success();
  }

  bool containsImpl(StringRef keyHash) const override {
    return cache.count(keyHash);
  }

  CacheFindResult findImpl(StringRef keyHash) const override {
    auto found = cache.find(keyHash);
    if (found == cache.end())
      return CacheFindResult::notInCache();

    // Create a memory buffer that holds this same data.
    return CacheFindResult::value((*found).second.copy());
  }

  ErrorOrSuccess clearImpl() override {
    cache.clear();
    return success();
  }

  llvm::StringMap<BufferRef> cache;
};
} // namespace

LLCL::RCRef<BlobCacheBackend>
M::Cache::getInMemoryBackend(LLCL::Runtime &runtime) {
  return LLCL::RCRef<InMemoryBackend>::create(runtime);
}

//===----------------------------------------------------------------------===//
// FilesystemBackend
//===----------------------------------------------------------------------===//

namespace {
/// Provides a filesystem-backed backend that stores the buffers in binary files
/// on disk.
struct FilesystemBackend : public BlobCacheBackend {
  explicit FilesystemBackend(LLCL::Runtime &runtime,
                             const std::filesystem::path &basePath)
      : BlobCacheBackend(runtime), basePath(basePath.string()) {}

  ErrorOrSuccess insertImpl(StringRef keyHash, BufferRef obj) override {
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
            os.write(obj->getBufferStart(), obj->getBufferSize());

            // Compute and copy the HMAC as well.
            SHA256Hash hash = hmacSHA256(obj->getBuffer(), kIntegrityKey);
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
    auto bufOr = Buffer::getFile(filePath);
    // If the file doesn't exist, or it's empty, return an error.
    if (failed(bufOr))
      return CacheFindResult::error(bufOr.takeError());

    BufferRef buffer = std::move(*bufOr);
    if (buffer->getBufferSize() == 0)
      return CacheFindResult::error("file '" + Twine(filePath.string()) +
                                    "' exists, but is empty");

    StringRef contentsAndHMAC = buffer->getBuffer();

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
    bufOr = Buffer::getFile(filePath, contents.size(),
                            /*Offset=*/0);
    if (failed(bufOr))
      return CacheFindResult::error(bufOr.takeError());
    // Otherwise, we're done.
    return CacheFindResult::value(std::move(*bufOr));
  }

  ErrorOrSuccess clearImpl() override {
    std::error_code ec;
    std::filesystem::remove_all(basePath, ec);
    if (ec)
      return Error(ec.message());

    return success();
  }

  std::filesystem::path getAbsolutePathForKey(StringRef keyHash) const {
    std::filesystem::path filepath(basePath);
    std::string encodedHash = llvm::encodeBase64(keyHash);
    std::replace_if(
        encodedHash.begin(), encodedHash.end(), [](char c) { return c == '/'; },
        '_');
    filepath /= encodedHash;

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

LLCL::RCRef<BlobCacheBackend>
M::Cache::getFilesystemBackend(LLCL::Runtime &runtime,
                               const std::filesystem::path &basePath) {
  return RCRef<FilesystemBackend>::create(runtime, basePath);
}

ErrorOr<LLCL::RCRef<BlobCacheBackend>>
M::Cache::getDefaultBackendChain(LLCL::Runtime &runtime,
                                 const std::filesystem::path &basePath) {
  auto backend = getInMemoryBackend(runtime);

  // Default to be in the `.derived` folder if we can.
  std::error_code ec;
  std::filesystem::path derived = std::filesystem::absolute(
      llvm::sys::Process::GetEnv("MODULAR_DERIVED_PATH").value_or("."), ec);
  if (ec)
    return Error(ec.message());

  std::filesystem::path base = basePath;
  if (!base.is_absolute())
    base = derived / basePath;

  // Erase everything that lives in basePath other than `base/version` if
  // we (a) have a `base`, (b) it exists, and (c) it's a directory.
  if (!base.empty() && std::filesystem::exists(base, ec) &&
      std::filesystem::is_directory(base, ec)) {

    // Iterate the base path and remove directories that don't match the
    // current version.
    for (const auto &dirEntry : std::filesystem::directory_iterator{base}) {
      // The directory entry must exist, be a directory, the parent must be
      // `base` and the directory 'filename' must not match
      // MODULAR_VERSION_STRING in order for it to be deleted.
      if (std::filesystem::is_directory(dirEntry.path(), ec) &&
          dirEntry.path().parent_path() == base &&
          dirEntry.path().filename() != MODULAR_VERSION_STRING) {
        std::filesystem::remove_all(dirEntry, ec);
      }
    }
  }

  base = base / MODULAR_VERSION_STRING;

  backend->setDelegate(getFilesystemBackend(runtime, base));
  return backend;
}
