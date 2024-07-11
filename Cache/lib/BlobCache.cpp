//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Cache/BlobCache.h"
#include "AsyncRT/Runtime/Algorithms.h"
#include "AsyncRT/Runtime/Runtime.h"
#include "AsyncRT/Support/UnknownLocationDecoder.h"
#include "Config/Version.h"
#include "Support/Base64.h"
#include "Support/Configuration.h"
#include "Support/FileSystemExtras.h"
#include "Support/Filesystem/DiskUsage.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/xxhash.h"
#include <shared_mutex>
#include <string_view>

using namespace M;
using namespace Cache;
using namespace AsyncRT;

/// Provides a simple way to get an error given an optional encoded location and
/// a standard Error.
static EncodedDiagnostic getError(std::optional<EncodedLocation> loc,
                                  Error err) {
  if (loc)
    return {std::move(err), std::move(*loc)};

  return UnknownLocationDecoder::getDiagnostic(std::move(err));
}

/// Returns whether the given path is a directory that the current process can
/// write to. If the path does not exist, this attempts to create a writable
/// directory at that path.
static bool checkOrCreateWriteableDirectory(const std::filesystem::path &path) {
  [[maybe_unused]] std::error_code existsErr;
  if (std::filesystem::exists(path, existsErr)) {
    // If the path exists but is not a directory, return false.
    if (!std::filesystem::is_directory(path, existsErr))
      return false;
    // Otherwise, check the write access permissions for the existing directory.
    return !llvm::sys::fs::access(path.string(),
                                  llvm::sys::fs::AccessMode::Write);
  }

  // If the path doesn't exist, create it. If creation was successful, we must
  // have write access.
  std::error_code createErr;
  std::filesystem::create_directories(path, createErr);
  return !createErr;
}

//===----------------------------------------------------------------------===//
// BlobCacheBackend
//===----------------------------------------------------------------------===//

AsyncValueRef<Chain>
BlobCacheBackend::insert(AsyncRT::Runtime &runtime, BufferRef keyHash,
                         BufferRef obj, std::optional<EncodedLocation> loc) {
  EncodedLocation location = loc.has_value()
                                 ? std::move(*loc)
                                 : UnknownLocationDecoder::getEncodedLocation();

  auto chain = insertImpl(runtime, keyHash.copy(), obj.copy(), location.copy());
  if (!delegate)
    return chain;

  // Arrange to wait and then insert into the delegate.
  auto result = AsyncValueRef<Chain>::allocate(runtime);
  std::move(chain).andThenSync(
      [result = result.copy(), thisRef = copyRCRef(this),
       keyHash = std::move(keyHash), obj = std::move(obj),
       loc = std::move(location)](AsyncValueRef<Chain> &&chain) mutable {
        if (chain.isError())
          return std::move(result).setToError(chain.takeDiagnostic());
        auto insert =
            thisRef->delegate->insert(result.getRuntime(), std::move(keyHash),
                                      std::move(obj), std::move(loc));
        std::move(insert).andThenSync(
            [result =
                 std::move(result)](AsyncValueRef<Chain> &&insert) mutable {
              if (insert.isError())
                return std::move(result).setToError(insert.takeDiagnostic());
              return std::move(result).emplace();
            });
      });
  return result;
}

ErrorOrSuccess BlobCacheBackend::insertSync(StringRef keyHash, BufferRef obj) {
  auto err = insertSyncImpl(keyHash, obj.copy());
  if (err.isError())
    return err.takeError();
  if (!delegate)
    return success();

  // Insert synchronously into the delegate as well.
  return delegate->insertSync(keyHash, std::move(obj));
}

AsyncValueRef<Chain>
BlobCacheBackend::insertImpl(AsyncRT::Runtime &runtime, BufferRef keyHash,
                             BufferRef obj,
                             std::optional<EncodedLocation> loc) {
  // Wrap the synchronous implementation by default.
  auto result = AsyncValueRef<Chain>::allocate(runtime);
  addTask(runtime, [thisRef = copyRCRef(this), keyHash = std::move(keyHash),
                    obj = std::move(obj), result = result.copy(),
                    loc = std::move(loc)]() mutable {
    if (auto err = thisRef->insertSync(keyHash->getBuffer(), std::move(obj))) {
      return std::move(result).setToError(
          getError(std::move(loc), err.takeError()));
    }
    std::move(result).emplace();
  });
  return result;
}

AsyncValueRef<bool>
BlobCacheBackend::contains(AsyncRT::Runtime &runtime, BufferRef keyHash,
                           std::optional<EncodedLocation> loc) {
  EncodedLocation location = loc.has_value()
                                 ? std::move(*loc)
                                 : UnknownLocationDecoder::getEncodedLocation();

  auto chain = containsImpl(runtime, keyHash.copy(), location.copy());
  if (!delegate)
    return chain;

  // Check the delegate, if this fails.
  auto result = AsyncValueRef<bool>::allocate(runtime);
  std::move(chain).andThenSync(
      [result = result.copy(), thisRef = copyRCRef(this),
       keyHash = std::move(keyHash),
       loc = std::move(location)](AsyncValueRef<bool> &&chain) mutable {
        if (chain.isError())
          return std::move(result).setToError(chain.takeDiagnostic());
        if (*chain)
          return std::move(result).emplace(true); // Value is locally available.
        // Need to schedule a delegate contains call.
        auto contains = thisRef->delegate->contains(
            result.getRuntime(), std::move(keyHash), std::move(loc));
        std::move(contains).andThenSync(
            [result =
                 std::move(result)](AsyncValueRef<bool> &&contains) mutable {
              if (contains.isError())
                return std::move(result).setToError(contains.takeDiagnostic());
              return std::move(result).emplace(*contains);
            });
      });
  return result;
}

ErrorOr<bool> BlobCacheBackend::containsSync(StringRef keyHash) {
  auto errOr = containsSyncImpl(keyHash);
  if (errOr.isError())
    return errOr.takeError();
  if (*errOr)
    return true;
  if (!delegate)
    return false;

  // Check the delegate synchronously.
  return delegate->containsSync(keyHash);
}

AsyncValueRef<bool>
BlobCacheBackend::containsImpl(AsyncRT::Runtime &runtime, BufferRef keyHash,
                               std::optional<EncodedLocation> loc) {
  // Wrap the synchronous implementation by default.
  auto result = AsyncValueRef<bool>::allocate(runtime);
  addTask(runtime, [thisRef = copyRCRef(this), keyHash = std::move(keyHash),
                    result = result.copy(), loc = std::move(loc)]() mutable {
    auto errOr = thisRef->containsSync(keyHash->getBuffer());
    if (errOr.isError()) {
      return std::move(result).setToError(
          getError(std::move(loc), errOr.takeError()));
    }
    std::move(result).emplace(*errOr);
  });
  return result;
}

AsyncValueRef<std::optional<BufferRef>>
BlobCacheBackend::find(AsyncRT::Runtime &runtime, BufferRef keyHash,
                       std::optional<EncodedLocation> loc) {
  EncodedLocation location = loc.has_value()
                                 ? std::move(*loc)
                                 : UnknownLocationDecoder::getEncodedLocation();

  auto chain = findImpl(runtime, keyHash.copy(), location.copy());
  if (!delegate)
    return chain;

  // Check the delegate, if this fails.
  auto result = AsyncValueRef<std::optional<BufferRef>>::allocate(runtime);
  std::move(chain).andThenSync(
      [result = result.copy(), thisRef = copyRCRef(this),
       keyHash = std::move(keyHash), loc = std::move(location)](
          AsyncValueRef<std::optional<BufferRef>> &&chain) mutable {
        if (chain.isError())
          return std::move(result).setToError(chain.takeDiagnostic());
        if (chain->has_value())
          return std::move(result).emplace(
              std::move(**chain)); // Found locally.
        // Need to attempt to find within the delegate.
        auto found = thisRef->delegate->find(result.getRuntime(),
                                             keyHash.copy(), loc.copy());
        std::move(found).andThenSync(
            [thisRef = thisRef.copy(), result = std::move(result),
             keyHash = std::move(keyHash), loc = std::move(loc)](
                AsyncValueRef<std::optional<BufferRef>> &&found) mutable {
              if (found.isError())
                return std::move(result).setToError(found.takeDiagnostic());
              if (!found->has_value())
                return std::move(result).emplace(std::nullopt); // Not found.
              // We need to insert locally.
              auto inserted =
                  thisRef->insert(result.getRuntime(), std::move(keyHash),
                                  (*found)->copy(), std::move(loc));
              std::move(inserted).andThenSync(
                  [result = std::move(result), obj = std::move(**found)](
                      AsyncValueRef<Chain> &&inserted) mutable {
                    if (inserted.isError())
                      return std::move(result).setToError(
                          inserted.takeDiagnostic());
                    return std::move(result).emplace(
                        std::move(obj)); // Finally, put the buffer.
                  });
            });
      });
  return result;
}

ErrorOr<std::optional<BufferRef>>
BlobCacheBackend::findSync(StringRef keyHash) {
  auto errOr = findSyncImpl(keyHash);
  if (errOr.isError())
    return errOr.takeError();
  if (errOr->has_value())
    return std::move(**errOr);
  if (!delegate)
    return std::nullopt;

  // Check the delegate synchronously.
  auto delegateErrOr = delegate->findSync(keyHash);
  if (delegateErrOr.isError())
    return delegateErrOr.takeError();
  if (!delegateErrOr->has_value())
    return std::nullopt;
  BufferRef buf = std::move(**delegateErrOr);

  // Insert the value locally.
  auto insertOr = insertSync(keyHash, buf.copy());
  if (insertOr.isError())
    return insertOr.takeError();

  // Return the loaded buffer.
  return std::move(buf);
}

AsyncValueRef<std::optional<BufferRef>>
BlobCacheBackend::findImpl(AsyncRT::Runtime &runtime, BufferRef keyHash,
                           std::optional<EncodedLocation> loc) {
  // Wrap the synchronous execution by default.
  auto result = AsyncValueRef<std::optional<BufferRef>>::allocate(runtime);
  addTask(runtime, [thisRef = copyRCRef(this), keyHash = std::move(keyHash),
                    result = result.copy(), loc = std::move(loc)]() mutable {
    auto errOr = thisRef->findSync(keyHash->getBuffer());
    if (errOr.isError())
      return std::move(result).setToError(
          getError(std::move(loc), errOr.takeError()));
    if (!errOr->has_value())
      return std::move(result).emplace(std::nullopt);
    BufferRef buf = std::move(**errOr);
    std::move(result).emplace(std::move(buf));
  });
  return result;
}

void BlobCacheBackend::appendDelegate(RCRef<BlobCacheBackend> d) {
  if (!delegate)
    delegate = std::move(d);
  else
    delegate->appendDelegate(std::move(d));
}

//===----------------------------------------------------------------------===//
// InMemoryBackend
//===----------------------------------------------------------------------===//

namespace {
/// Provides an in-memory backend that stores memory buffers in an
/// llvm::StringMap.
struct InMemoryBackend : public BlobCacheBackend {
  ErrorOrSuccess insertSyncImpl(StringRef keyHash, BufferRef obj) override {
    std::lock_guard<std::shared_mutex> lock(mutex);
    cache[keyHash] = std::move(obj);
    return success();
  }

  ErrorOr<bool> containsSyncImpl(StringRef keyHash) override {
    std::shared_lock<std::shared_mutex> lock(mutex);
    return cache.count(keyHash);
  }

  ErrorOr<std::optional<BufferRef>> findSyncImpl(StringRef keyHash) override {
    std::shared_lock<std::shared_mutex> lock(mutex);
    auto found = cache.find(keyHash);
    if (found == cache.end())
      return std::nullopt;
    return found->second.copy();
  }

  llvm::StringMap<BufferRef> cache;
  mutable std::shared_mutex mutex;
};
} // namespace

RCRef<BlobCacheBackend> M::Cache::getInMemoryBackend() {
  return RCRef<InMemoryBackend>::create();
}

//===----------------------------------------------------------------------===//
// FilesystemBackend
//===----------------------------------------------------------------------===//

namespace {
/// Provides a filesystem-backed backend that primarily stores the buffers in
/// binary files on disk. If read-only, no writes are performed, only reads.
struct FilesystemBackend : public BlobCacheBackend {
  explicit FilesystemBackend(const std::filesystem::path &basePath,
                             bool readOnly)
      : basePath(basePath.string()), readOnly(readOnly) {}

  ErrorOrSuccess insertSyncImpl(StringRef keyHash, BufferRef obj) override {
    // Check if we already have the object in the filesystem cache - if we do,
    // then don't bother writing it again.
    ErrorOr<bool> containsOr = containsSync(keyHash);
    if (!containsOr.isError() && *containsOr)
      return success();

    // Otherwise, if the filesystem is read-only, we cannot write to it for
    // insertion.
    if (readOnly)
      return success();

    // Get the absolute path and create any directories we need to create.
    ErrorOr<std::filesystem::path> filePathOr = getAbsolutePathForKey(keyHash);
    if (filePathOr.isError())
      return filePathOr.takeError();

    std::error_code dirErr;
    std::filesystem::create_directories(filePathOr->parent_path(), dirErr);
    if (dirErr)
      return Error(dirErr.message());

    auto availableSizeOr = M::getAvailableDiskSpace(filePathOr->parent_path());
    if (availableSizeOr.isError())
      return availableSizeOr.takeError();

    if (*availableSizeOr < obj->getBufferSize())
      return Error("cannot write to file to filesystem cache since available "
                   "space(" +
                   std::to_string(*availableSizeOr) +
                   ") is not enough to write " +
                   std::to_string(obj->getBufferSize()) + " bytes");

    // Functor used when we actually need to write out the file.
    auto writeContent = [&](raw_ostream &os) {
      // Copy the data into the file buffer.
      os.write(obj->getBufferStart(), obj->getBufferSize());

      // Compute and copy the hash as well.
      llvm::XXH128_hash_t hash =
          llvm::xxh3_128bits(arrayRefFromStringRef(obj->getBuffer()));
      os.write(llvm::bit_cast<char *>(&hash), sizeof(llvm::XXH128_hash_t));
    };

    // Safely process creating the file, taking into account that we may
    // have different processes trying to produce this file in parallel.
    if (auto err = writeFileUnderLock(*filePathOr, writeContent); err.isError())
      return err.takeError();

    return success();
  }

  ErrorOr<bool> containsSyncImpl(StringRef keyHash) override {
    std::error_code ec;
    ErrorOr<std::filesystem::path> absOr = getAbsolutePathForKey(keyHash);
    if (absOr.isError())
      return absOr.takeError();

    bool exists = std::filesystem::exists(*absOr, ec);
    if (ec)
      return Error(ec.message());

    if (!exists)
      return false;

    bool is_dir = std::filesystem::is_directory(*absOr, ec);
    if (ec)
      return Error(ec.message());
    return !is_dir;
  }

  ErrorOr<std::optional<BufferRef>> findSyncImpl(StringRef keyHash) override {
    // Get the file path and open it.
    ErrorOr<std::filesystem::path> filePath = getAbsolutePathForKey(keyHash);

    // No such file, return nullopt (not error).
    std::error_code ec;
    if (!std::filesystem::exists(*filePath, ec) || ec)
      return std::nullopt;

    // If the cache file for this key exists, then it will never be written
    // again. We can safely read it without a lock.
    auto bufOr = Buffer::getFile(*filePath);
    // If the file doesn't exist, or it's empty, return an error.
    if (bufOr.isError())
      return bufOr.takeError();

    BufferRef buffer = std::move(*bufOr);
    if (buffer->getBufferSize() == 0)
      return Error("file '" + Twine(filePath->string()) +
                   "' exists, but is empty");

    StringRef contentsAndHash = buffer->getBuffer();

    // Get a StringRef of the contents without the hash.
    StringRef contents = contentsAndHash.drop_back(sizeof(llvm::XXH128_hash_t));
    llvm::XXH128_hash_t computedHash =
        llvm::xxh3_128bits(arrayRefFromStringRef(contents));
    StringRef storedHash =
        contentsAndHash.take_back(sizeof(llvm::XXH128_hash_t));

    // Check the computed hash against the hash in the file.
    if (memcmp(llvm::bit_cast<char *>(&computedHash), storedHash.data(),
               sizeof(llvm::XXH128_hash_t)) != 0) {
      return Error("corrupted file: stored hash and computed hash did not "
                   "match for file '" +
                   Twine(filePath->string()) + "'");
    }

    // Now that we've verified the integrity of the file, return a memory buffer
    // that holds just the contents.
    bufOr = Buffer::getFile(*filePath, contents.size(), /*offset=*/0);
    if (failed(bufOr))
      return bufOr.takeError();
    return BufferRef::take(bufOr->release());
  }

  ErrorOr<std::filesystem::path>
  getAbsolutePathForKey(StringRef keyHash) const {
    std::error_code ec;
    std::filesystem::path filepath(basePath);
    std::string encodedHash = encodeURLSafeBase64(keyHash);
    filepath /= encodedHash;

    std::filesystem::path absolute = std::filesystem::absolute(filepath, ec);
    if (ec)
      return Error(ec.message());

    return absolute;
  }

  /// The base path for the filesystem cache.
  std::string basePath;
  /// Whether the filesystem cache is read-only. If `true`, reads are performed
  /// as normal, whereas writes are silently ignored.
  bool readOnly;
};
} // namespace

RCRef<BlobCacheBackend>
M::Cache::getFilesystemBackend(const std::filesystem::path &basePath,
                               bool readOnly) {
  return RCRef<FilesystemBackend>::create(basePath, readOnly);
}

/// Returns a filesystem-based implementation of the BlobCacheBackend. The
/// `cacheDir` is used to derive a path for use by the filesystem backend. The
/// `version` specifies the version string of the cache, defaults to
/// MODULAR_VERSION_STRING if the provided version is empty.
static ErrorOr<RCRef<FilesystemBackend>>
getVersionedFilesystemBackend(const std::filesystem::path &cacheDir,
                              std::string_view version) {
  // If no version is specified, use the default version.
  if (version.empty())
    version = getModularVersionString();

  std::error_code ec;
  std::filesystem::path base = cacheDir;
  if (!base.is_absolute()) {
    // Default to the .derived directory.
    if (auto path = llvm::sys::Process::GetEnv("MODULAR_DERIVED_PATH")) {
      base = std::filesystem::absolute(*path, ec) / cacheDir;
      if (ec)
        return Error("failed to get absolute path to derived dir: " +
                     ec.message());
    } else if (auto path = llvm::sys::Process::GetEnv("MODULAR_INSTALL_DIR")) {
      base = std::filesystem::absolute(*path, ec) / cacheDir;
      if (ec)
        return Error("failed to get absolute path to installed dir: " +
                     ec.message());
    } else {
      // Attempt to find an existing home directory, but *do not create the
      // directory*. This is because we will fall back to the using the
      // hard-coded derived dir below. This logic should likely be reconciled
      // more generally, as it is generally used only for CI and local
      // development, which is a strange pattern.
      auto homePathOr = Config::getModularDataFolderPath(/*create=*/false);
      if (!homePathOr.isError() && std::filesystem::exists(*homePathOr, ec) &&
          !ec) {
        base = std::filesystem::absolute(*homePathOr, ec) / cacheDir;
        if (ec)
          return Error("failed to get absolute path to modular home dir: " +
                       ec.message());
#ifdef _WIN32
      } else if (auto path = findDirInEnvPath(cacheDir.string(), "PATH", ';')) {
#else
      } else if (auto path = findDirInEnvPath(cacheDir.string())) {
#endif
        base = std::filesystem::absolute(*path, ec);
        if (ec)
          return Error(
              "failed to get absolute path to directory specified by " + *path +
              ec.message());
      } else {
        auto derivedPath = std::filesystem::path(MODULAR_DERIVED_DIR);
        if (std::filesystem::exists(derivedPath, ec) && !ec)
          base = std::filesystem::absolute(derivedPath, ec) / cacheDir;
        else
          base = std::filesystem::temp_directory_path(ec) / cacheDir;
        if (ec)
          return Error("failed to get absolute path to derived dir: " +
                       ec.message());
      }
    }
  }

  assert(base.is_absolute() && "must default to non-empty absolute path");
  bool readOnly = !checkOrCreateWriteableDirectory(base);

  // If we have write access, do a little cache pruning on the host system in
  // order to keep disk usage down: iterate the base path and remove directories
  // that don't match the current version.
  if (!readOnly) {
    for (const auto &dirEntry : std::filesystem::directory_iterator{base}) {
      // The directory entry must exist, be a directory, the parent must be
      // `base` and the directory 'filename' must not match
      // MODULAR_VERSION_STRING in order for it to be deleted.

      [[maybe_unused]] std::error_code ec0, ec1;
      if (std::filesystem::is_directory(dirEntry.path(), ec) &&
          (std::filesystem::canonical(dirEntry.path().parent_path(), ec0) ==
           std::filesystem::canonical(base, ec1)) &&
          (dirEntry.path().filename() != version)) {
        std::filesystem::remove_all(dirEntry, ec);
      }
    }
  }

  base = base / version;
  return RCRef<FilesystemBackend>::create(base, readOnly);
}

ErrorOr<RCRef<BlobCacheBackend>>
M::Cache::getFilesystemBackend(const std::filesystem::path &cacheDir,
                               std::string_view version) {
  return getVersionedFilesystemBackend(cacheDir, version);
}

//===----------------------------------------------------------------------===//
// DylibBlobCacheBackend
//===----------------------------------------------------------------------===//

namespace {
/// This stub loads a shared library that implements a BlobCacheBackend
/// using the DylibBlobCacheBackend interface, and delegates backend calls
/// to this implementation.
struct DylibBackendStub : public BlobCacheBackend {

  static ErrorOr<RCRef<BlobCacheBackend>>
  create(StringRef libPath, const DylibBackendConfig *config) {
    auto backendStub = RCRef<DylibBackendStub>::create();
    if (auto err = backendStub->load(libPath, config))
      return err.takeError();
    return backendStub;
  }

  ~DylibBackendStub() override {
    // Release reference to ensure the backend is deleted before closing the
    // library.
    backend.reset();
    llvm::sys::DynamicLibrary::closeLibrary(dylib);
  }

  AsyncValueRef<Chain> insertImpl(AsyncRT::Runtime &runtime, BufferRef keyHash,
                                  BufferRef obj,
                                  std::optional<EncodedLocation> loc) override {
    return backend->insertImpl(runtime, std::move(keyHash), std::move(obj),
                               std::move(loc));
  }
  ErrorOrSuccess insertSyncImpl(StringRef keyHash, BufferRef obj) override {
    return backend->insertSyncImpl(keyHash, std::move(obj));
  }

  AsyncValueRef<bool>
  containsImpl(AsyncRT::Runtime &runtime, BufferRef keyHash,
               std::optional<EncodedLocation> loc) override {
    return backend->containsImpl(runtime, std::move(keyHash), std::move(loc));
  }
  ErrorOr<bool> containsSyncImpl(StringRef keyHash) override {
    return backend->containsSyncImpl(keyHash);
  }

  AsyncValueRef<std::optional<BufferRef>>
  findImpl(AsyncRT::Runtime &runtime, BufferRef keyHash,
           std::optional<EncodedLocation> loc) override {
    return backend->findImpl(runtime, std::move(keyHash), std::move(loc));
  }
  ErrorOr<std::optional<BufferRef>> findSyncImpl(StringRef keyHash) override {
    return backend->findSyncImpl(keyHash);
  }

  void appendDelegate(RCRef<BlobCacheBackend> d) override {
    return backend->appendDelegate(std::move(d));
  }

private:
  /// So RCRef can access private constructor.
  friend class RCRef<DylibBackendStub>;

  ErrorOrSuccess load(StringRef libPath, const DylibBackendConfig *config) {
    std::string errorMsg;
    dylib =
        llvm::sys::DynamicLibrary::getLibrary(libPath.str().c_str(), &errorMsg);
    if (!dylib.isValid())
      return Error("Failed to load library " + libPath + ": " + errorMsg);

    using allocType = DylibBlobCacheBackend *(*)();
    auto allocFunc = reinterpret_cast<allocType>(
        dylib.getAddressOfSymbol("M_CAS_allocateBackend"));
    if (!allocFunc) {
      llvm::sys::DynamicLibrary::closeLibrary(dylib);
      return Error("M_CAS_allocateBackend symbol not found\n");
    }
    backend = RCRef<DylibBlobCacheBackend>::take(allocFunc());
    return backend->setConfig(config);
  }

  /// The dynamic library handle.
  llvm::sys::DynamicLibrary dylib;
  /// The stub delegates all cache-related operations to this backend.
  RCRef<DylibBlobCacheBackend> backend;
};
} // namespace

ErrorOr<RCRef<BlobCacheBackend>>
M::Cache::getLocalDefaultBackendChain(const std::filesystem::path &cacheDir,
                                      std::string_view version) {
  auto filesystemOr = getFilesystemBackend(cacheDir, version);
  if (filesystemOr.isError())
    return filesystemOr.takeError();
  auto memory = getInMemoryBackend();
  memory->appendDelegate(std::move(*filesystemOr));
  return std::move(memory);
}

ErrorOr<RCRef<BlobCacheBackend>>
M::Cache::getDefaultBackendChain(const URI &uri, std::string_view version) {
  StringRef scheme = uri.getScheme();
  if (scheme == "file") {
    std::string path(uri.getPath());
    return getLocalDefaultBackendChain(path, version);
  }

  return Error("Can't build BlobCache backend chain with unknown URI scheme: " +
               scheme);
}
