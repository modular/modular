//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Cache/BlobCache.h"
#include "Config/Config.h"
#include "LLCL/Runtime/Algorithms.h"
#include "LLCL/Runtime/Runtime.h"
#include "LLCL/Support/UnknownLocationDecoder.h"
#include "Support/HMAC.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/LockFileManager.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"
#include <shared_mutex>

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
                    obj = obj.copy(), result = result.copy()]() mutable {
    if (auto err = thisRef->insertImpl(keyHash->getBuffer(), obj.copy()))
      return std::move(result).emplace(err.takeError());

    if (!thisRef->delegate)
      return std::move(result).emplace(success());

    auto insert = thisRef->delegate->insert(keyHash.copy(), obj.copy());
    std::move(insert).andThenSync(
        [thisRef = thisRef.copy(),
         // Safe to move local copy of result.
         result = std::move(result)](
            const AsyncValueRef<ErrorOrSuccess> &&insert) mutable {
          if (failed(*insert))
            return std::move(result).emplace(insert->takeError());
          std::move(result).emplace(success());
        });
  });
  return result;
}

namespace {
/// Provides a simple callable that encapsulates an optional EncodedLocation and
/// returns an EncodedDiagnostic of the correct kind based on if the location
/// exists or not.
struct GetError {
  std::optional<EncodedLocation> loc;

  EncodedDiagnostic operator()(Error err) {
    if (loc)
      return {std::move(err), std::move(*loc)};

    return UnknownLocationDecoder::getDiagnostic(std::move(err));
  }
};
} // namespace

AsyncValueRef<bool>
BlobCacheBackend::contains(BufferRef keyHash,
                           std::optional<EncodedLocation> loc) {
  auto result = AsyncValueRef<bool>::allocate(runtime);
  addTask(runtime, [thisRef = copyRCRef(this), keyHash = keyHash.copy(),
                    result = result.copy(),
                    getError = GetError{std::move(loc)}]() mutable {
    auto containsOr = thisRef->containsImpl(keyHash->getBuffer());
    if (containsOr.isError())
      return std::move(result).setToError(getError(containsOr.takeError()));

    if (*containsOr)
      return std::move(result).emplace(true);

    if (!thisRef->delegate)
      return std::move(result).emplace(false);

    auto contains = thisRef->delegate->contains(keyHash.copy());
    std::move(contains).andThenSync(
        [thisRef = thisRef.copy(),
         // Safe to move local copy of result.
         result = std::move(result)](AsyncValueRef<bool> &&contains) mutable {
          return std::move(result).emplace(*contains);
        });
  });
  return result;
}

AsyncValueRef<std::optional<BufferRef>>
BlobCacheBackend::find(BufferRef keyHash, std::optional<EncodedLocation> loc) {
  auto result = AsyncValueRef<std::optional<BufferRef>>::allocate(runtime);
  addTask(runtime, [thisRef = copyRCRef(this), keyHash = keyHash.copy(),
                    result = result.copy(),
                    getError = GetError{std::move(loc)}]() mutable {
    // Check if the key is contained at this cache level (return an error if
    // containsOr returns an error).
    auto containsOr = thisRef->containsImpl(keyHash->getBuffer());
    if (containsOr.isError())
      return std::move(result).setToError(getError(containsOr.takeError()));

    // If we have it, return it and don't bother delegating.
    if (*containsOr) {
      // If this was an error, return that error in the AsyncValue.
      auto bufOr = thisRef->findImpl(keyHash->getBuffer());
      if (bufOr.isError())
        return std::move(result).setToError(getError(bufOr.takeError()));

      // Otherwise, simply return the contents.
      return std::move(result).emplace(std::move(*bufOr));
    }

    if (!thisRef->delegate)
      return std::move(result).emplace(std::nullopt);

    auto itemOr = thisRef->delegate->find(keyHash.copy());
    std::move(itemOr).andThenSync(
        [thisRef = thisRef.copy(), keyHash = keyHash.copy(),
         getError = std::move(getError),
         // Safe to move local copy of result.
         result = std::move(result)](
            AsyncValueRef<std::optional<BufferRef>> &&itemOr) mutable {
          if (itemOr.isError())
            return std::move(result).setToError(itemOr.takeDiagnostic());

          // Delegate doesn't have it either!
          if (!*itemOr)
            return std::move(result).emplace(std::nullopt);

          BufferRef item = std::move(**itemOr);

          // Store the item in our cache level so we can get a cache hit
          // later.
          if (auto err = thisRef->insertImpl(keyHash->getBuffer(), item.copy()))
            return std::move(result).setToError(getError(err.takeError()));

          // Return the item.
          return std::move(result).emplace(std::move(item));
        });
  });

  return result;
}

AsyncValueRef<ErrorOrSuccess> BlobCacheBackend::clear() {
  auto result = AsyncValueRef<ErrorOrSuccess>::allocate(runtime);
  addTask(runtime,
          [thisRef = copyRCRef(this), result = result.copy()]() mutable {
            if (auto err = thisRef->clearImpl())
              return std::move(result).emplace(err.takeError());

            if (!thisRef->delegate)
              return std::move(result).emplace(success());

            auto clear = thisRef->delegate->clear();
            std::move(clear).andThenSync(
                [thisRef = thisRef.copy(), result = result.copy()](
                    AsyncValueRef<ErrorOrSuccess> &&clear) mutable {
                  std::move(result).emplace(std::move(*clear));
                });
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
    std::lock_guard<std::shared_mutex> lock(mutex);

    // Store the item in this cache.
    cache[keyHash] = std::move(obj);
    return success();
  }

  ErrorOr<bool> containsImpl(StringRef keyHash) const override {
    std::shared_lock<std::shared_mutex> lock(mutex);
    return cache.count(keyHash);
  }

  ErrorOr<std::optional<BufferRef>> findImpl(StringRef keyHash) const override {
    std::shared_lock<std::shared_mutex> lock(mutex);
    auto found = cache.find(keyHash);
    if (found == cache.end())
      return std::nullopt;

    // Get a copy of the buffer that holds this data.
    return (*found).second.copy();
  }

  ErrorOrSuccess clearImpl() override {
    std::lock_guard<std::shared_mutex> lock(mutex);
    cache.clear();
    return success();
  }

  llvm::StringMap<BufferRef> cache;
  mutable std::shared_mutex mutex;
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
            BLAKE3Hash hash = hmacBLAKE3(obj->getBuffer(), kIntegrityKey);
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

  ErrorOr<bool> containsImpl(StringRef keyHash) const override {
    auto abs = getAbsolutePathForKey(keyHash);
    return std::filesystem::exists(abs) && !std::filesystem::is_directory(abs);
  }

  ErrorOr<std::optional<BufferRef>> findImpl(StringRef keyHash) const override {
    // Get the file path and open it.
    std::filesystem::path filePath = getAbsolutePathForKey(keyHash);
    auto bufOr = Buffer::getFile(filePath);
    // If the file doesn't exist, or it's empty, return an error.
    if (failed(bufOr))
      return bufOr.takeError();

    BufferRef buffer = std::move(*bufOr);
    if (buffer->getBufferSize() == 0)
      return Error("file '" + Twine(filePath.string()) +
                   "' exists, but is empty");

    StringRef contentsAndHMAC = buffer->getBuffer();

    // Get a StringRef of the contents without the HMAC.
    StringRef contents = contentsAndHMAC.drop_back(blake3Bytes);
    BLAKE3Hash computedHMAC = hmacBLAKE3(contents, kIntegrityKey);
    StringRef storedHMAC = contentsAndHMAC.take_back(blake3Bytes);

    // Check the computed hmac against the one in the file.
    if (memcmp(computedHMAC.data(), storedHMAC.data(), blake3Bytes)) {
      return Error("corrupted file: stored hash and computed hash did not "
                   "match for file '" +
                   Twine(filePath.string()) + "'");
    }

    // Now that we've verified the integrity of the file, return a memory buffer
    // that holds just the contents.
    bufOr = Buffer::getFile(filePath, contents.size(),
                            /*offset=*/0);
    if (failed(bufOr))
      return bufOr.takeError();
    // Otherwise, we're done.
    return std::move(*bufOr);
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
                                 const std::filesystem::path &cacheDir) {
  auto backend = getInMemoryBackend(runtime);

  // Default to be in the `.derived` folder if we can.
  std::error_code ec;
  std::filesystem::path derived = std::filesystem::absolute(
      llvm::sys::Process::GetEnv("MODULAR_DERIVED_PATH").value_or("."), ec);
  if (ec)
    return Error(ec.message());

  std::filesystem::path base = cacheDir;
  if (!base.is_absolute())
    base = derived / cacheDir;

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
