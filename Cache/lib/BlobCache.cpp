//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Cache/BlobCache.h"
#include "Config/Version.h"
#include "LLCL/Runtime/Algorithms.h"
#include "LLCL/Runtime/Runtime.h"
#include "LLCL/Support/UnknownLocationDecoder.h"
#include "Support/Base64.h"
#include "Support/Configuration.h"
#include "Support/FileSystemExtras.h"
#include "Support/Filesystem/DiskUsage.h"
#include "Support/HMAC.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"
#include <shared_mutex>

using namespace M;
using namespace Cache;
using namespace LLCL;

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
static bool checkOrCreateWriteableDirectory(std::filesystem::path path) {
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
BlobCacheBackend::insert(BufferRef keyHash, BufferRef obj,
                         std::optional<EncodedLocation> loc) {
  auto result = AsyncValueRef<Chain>::allocate(runtime);
  addTask(runtime, [thisRef = copyRCRef(this), keyHash = keyHash.copy(),
                    obj = obj.copy(), result = result.copy(),
                    loc = std::move(loc)]() mutable {
    if (auto err = thisRef->insertImpl(keyHash->getBuffer(), obj.copy())) {
      return std::move(result).setToError(
          getError(std::move(loc), err.takeError()));
    }

    return thisRef->delegateInsert(std::move(result), std::move(keyHash),
                                   std::move(obj), std::move(loc));
  });
  return result;
}

void BlobCacheBackend::delegateInsert(
    AsyncValueRef<Chain> result, BufferRef keyHash, BufferRef obj,
    std::optional<LLCL::EncodedLocation> loc) {
  if (!delegate)
    return std::move(result).emplace();

  AsyncValueRef<Chain> insert =
      delegate->insert(keyHash.copy(), obj.copy(), std::move(loc));
  std::move(insert).andThenSync(
      [thisRef = copyRCRef(this),
       // Safe to move local copy of result.
       result = std::move(result)](AsyncValueRef<Chain> &&insert) mutable {
        if (insert.isError())
          return std::move(result).setToError(insert.takeDiagnostic());

        return std::move(result).emplace();
      });
}

AsyncValueRef<bool>
BlobCacheBackend::contains(BufferRef keyHash,
                           std::optional<EncodedLocation> loc) {
  auto result = AsyncValueRef<bool>::allocate(runtime);
  addTask(runtime, [thisRef = copyRCRef(this), keyHash = keyHash.copy(),
                    result = result.copy(), loc = std::move(loc)]() mutable {
    auto containsOr = thisRef->containsImpl(keyHash->getBuffer());
    if (containsOr.isError()) {
      return std::move(result).setToError(
          getError(std::move(loc), containsOr.takeError()));
    }

    if (*containsOr)
      return std::move(result).emplace(true);

    return thisRef->delegateContains(std::move(result), std::move(keyHash),
                                     std::move(loc));
  });
  return result;
}

void BlobCacheBackend::delegateContains(
    AsyncValueRef<bool> result, BufferRef keyHash,
    std::optional<LLCL::EncodedLocation> loc) {
  if (!delegate)
    return std::move(result).emplace(false);

  auto contains = delegate->contains(keyHash.copy(), std::move(loc));
  std::move(contains).andThenSync(
      [thisRef = copyRCRef(this),
       // Safe to move local copy of result.
       result = std::move(result)](AsyncValueRef<bool> &&contains) mutable {
        if (contains.isError())
          return std::move(result).setToError(contains.takeDiagnostic());

        return std::move(result).emplace(*contains);
      });
}

AsyncValueRef<std::optional<BufferRef>>
BlobCacheBackend::find(BufferRef keyHash,
                       std::optional<WriteableBufferRef> backingBuf,
                       std::optional<EncodedLocation> loc) {
  auto result = AsyncValueRef<std::optional<BufferRef>>::allocate(runtime);
  addTask(runtime, [thisRef = copyRCRef(this), keyHash = keyHash.copy(),
                    result = result.copy(), backingBuf = std::move(backingBuf),
                    loc = std::move(loc)]() mutable {
    // Find it at this level.
    ErrorOr<std::optional<BufferRef>> bufOr = thisRef->findImpl(
        keyHash->getBuffer(),
        (backingBuf ? std::optional<WriteableBufferRef>(backingBuf->copy())
                    : std::nullopt));
    if (bufOr.isError()) {
      return std::move(result).setToError(
          getError(std::move(loc), bufOr.takeError()));
    }

    // If we had it, return, and we're done.
    if (bufOr->has_value())
      return std::move(result).emplace(std::move(*bufOr));

    // If we don't have it, try with delegate.
    return thisRef->delegateFind(std::move(result), std::move(keyHash),
                                 std::move(backingBuf), std::move(loc));
  });

  return result;
}

void BlobCacheBackend::delegateFind(
    AsyncValueRef<std::optional<BufferRef>> result, BufferRef keyHash,
    std::optional<WriteableBufferRef> backingBuf,
    std::optional<EncodedLocation> loc) {
  // No delegate and we don't have it, return nullopt.
  if (!delegate)
    return std::move(result).emplace(std::nullopt);

  // Create a concrete location we can use here - we always need *a* location,
  // even if it's unknown.
  EncodedLocation location = loc.has_value()
                                 ? std::move(*loc)
                                 : UnknownLocationDecoder::getEncodedLocation();

  auto itemOr = delegate->find(
      keyHash.copy(),
      (backingBuf ? std::optional<WriteableBufferRef>(backingBuf->copy())
                  : std::nullopt),
      location.copy());
  std::move(itemOr).andThenSync(
      [thisRef = copyRCRef(this), keyHash = keyHash.copy(),
       location = std::move(location),
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
        if (auto err = thisRef->insertImpl(keyHash->getBuffer(), item.copy())) {
          return std::move(result).setToError(
              {err.takeError(), std::move(location)});
        }

        // Return the item.
        return std::move(result).emplace(std::move(item));
      });
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

  ErrorOr<std::optional<BufferRef>>
  findImpl(StringRef keyHash,
           std::optional<WriteableBufferRef> backingBuf) const override {
    std::shared_lock<std::shared_mutex> lock(mutex);
    auto found = cache.find(keyHash);
    if (found == cache.end())
      return std::nullopt;

    // No buffer provided, give back a ref to the buffer we have.
    if (!backingBuf)
      return found->second.copy();

    // If we were passed in a buffer...
    Buffer &foundBuf = *found->second;
    // If the buffer already contains the data, don't bother doing anything.
    if ((*backingBuf)->getBufferStart() == foundBuf.getBufferStart())
      return found->second.copy();

    if ((*backingBuf)->getBufferCapacity() < foundBuf.getBufferSize()) {
      return Error("Buffer passed to CAS (size " +
                   Twine((*backingBuf)->getBufferCapacity()) +
                   ") cannot accommodate found object (size " +
                   Twine(foundBuf.getBufferSize()) + ")");
    }

    // Write the contents into the buffer. The buffer may have data inside it
    // already, so we have to get the starting offset.
    uint64_t startOffset = (*backingBuf)->tell();
    (*backingBuf)->write(foundBuf.getBufferStart(), foundBuf.getBufferSize());
    // And return an alias to *that* buffer.
    return Buffer::getAlias(backingBuf->copy(), startOffset,
                            foundBuf.getBufferSize());
  }

  llvm::StringMap<BufferRef> cache;
  mutable std::shared_mutex mutex;
};
} // namespace

RCRef<BlobCacheBackend> M::Cache::getInMemoryBackend(LLCL::Runtime &runtime) {
  return RCRef<InMemoryBackend>::create(runtime);
}

//===----------------------------------------------------------------------===//
// FilesystemBackend
//===----------------------------------------------------------------------===//

namespace {
/// Provides a filesystem-backed backend that primarily stores the buffers in
/// binary files on disk. If read-only, no writes are performed, only reads.
struct FilesystemBackend : public BlobCacheBackend {
  explicit FilesystemBackend(LLCL::Runtime &runtime,
                             const std::filesystem::path &basePath,
                             bool readOnly)
      : BlobCacheBackend(runtime), basePath(basePath.string()),
        readOnly(readOnly) {}

  ErrorOrSuccess insertImpl(StringRef keyHash, BufferRef obj) override {
    // Check if we already have the object in the filesystem cache - if we do,
    // then don't bother writing it again.
    ErrorOr<bool> containsOr = containsImpl(keyHash);
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

      // Compute and copy the HMAC as well.
      BLAKE3Hash hash = hmacBLAKE3(obj->getBuffer(), kIntegrityKey);
      os.write((const char *)hash.data(), hash.size());
    };

    // Safely process creating the file, taking into account that we may
    // have different processes trying to produce this file in parallel.
    if (auto err = writeFileUnderLock(*filePathOr, writeContent); err.isError())
      return err.takeError();

    return success();
  }

  ErrorOr<bool> containsImpl(StringRef keyHash) const override {
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

  ErrorOr<std::optional<BufferRef>>
  findImpl(StringRef keyHash,
           std::optional<WriteableBufferRef> backingBuf) const override {
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

    StringRef contentsAndHMAC = buffer->getBuffer();

    // Get a StringRef of the contents without the HMAC.
    StringRef contents = contentsAndHMAC.drop_back(blake3Bytes);
    BLAKE3Hash computedHMAC = hmacBLAKE3(contents, kIntegrityKey);
    StringRef storedHMAC = contentsAndHMAC.take_back(blake3Bytes);

    // Check the computed hmac against the one in the file.
    if (memcmp(computedHMAC.data(), storedHMAC.data(), blake3Bytes) != 0) {
      return Error("corrupted file: stored hash and computed hash did not "
                   "match for file '" +
                   Twine(filePath->string()) + "'");
    }

    // Now that we've verified the integrity of the file, return a memory buffer
    // that holds just the contents.
    bufOr = Buffer::getFile(*filePath, contents.size(), /*offset=*/0);
    if (failed(bufOr))
      return bufOr.takeError();

    // No buffer provided, return the mapped thing directly.
    if (!backingBuf)
      return BufferRef::take(bufOr->release());

    if ((*backingBuf)->getBufferCapacity() < (*bufOr)->getBufferSize()) {
      return Error("Buffer passed to CAS (size " +
                   Twine((*backingBuf)->getBufferCapacity()) +
                   ") cannot accommodate found object (size " +
                   Twine((*bufOr)->getBufferSize()) + ")");
    }

    // Copy the file data into the buffer that was provided to us.
    uint64_t startOffset = (*backingBuf)->tell();
    (*backingBuf)->write((*bufOr)->getBufferStart(), (*bufOr)->getBufferSize());
    // Take an alias to the provided buffer and return it.
    return Buffer::getAlias(backingBuf->copy(), startOffset,
                            (*bufOr)->getBufferSize());
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

  /// This is a CSPRNG-generated 32-byte string. It's used for integrity
  /// checking in the HMAC.
  static constexpr llvm::StringLiteral kIntegrityKey =
      "bedcaea9f09fa9fe565a8088ea66547c06c7c8e9c47fa46e0fb768a157d640a6";
  /// The base path for the filesystem cache.
  std::string basePath;
  /// Whether the filesystem cache is read-only. If `true`, reads are performed
  /// as normal, whereas writes are silently ignored.
  bool readOnly;
};
} // namespace

RCRef<BlobCacheBackend>
M::Cache::getFilesystemBackend(LLCL::Runtime &runtime,
                               const std::filesystem::path &basePath,
                               bool readOnly) {
  return RCRef<FilesystemBackend>::create(runtime, basePath, readOnly);
}

/// Returns a filesystem-based implementation of the BlobCacheBackend. The
/// `cacheDir` is used to derive a path for use by the filesystem backend. The
/// `version` specifies the version string of the cache, defaults to
/// MODULAR_VERSION_STRING if the provided version is empty.
static ErrorOr<RCRef<FilesystemBackend>>
getVersionedFilesystemBackend(LLCL::Runtime &runtime,
                              const std::filesystem::path &cacheDir,
                              std::string version) {
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
  return RCRef<FilesystemBackend>::create(runtime, base, readOnly);
}

ErrorOr<RCRef<BlobCacheBackend>>
M::Cache::getFilesystemBackend(LLCL::Runtime &runtime,
                               const std::filesystem::path &cacheDir,
                               const std::string &version) {
  return getVersionedFilesystemBackend(runtime, cacheDir, version);
}
//===----------------------------------------------------------------------===//
// FileSystemBackedInMemoryBackend
//===----------------------------------------------------------------------===//

namespace {
/// Provides a wrapper around in-memory and filesystem-backed backends that only
/// stores mmap'd buffers in-memory. This is useful for caching large objects
/// or large numbers of objects that would otherwise consume too much memory.
struct FileSystemBackedInMemoryBackend : public BlobCacheBackend {
  explicit FileSystemBackedInMemoryBackend(
      LLCL::Runtime &runtime, RCRef<InMemoryBackend> inmemoryBackend,
      RCRef<FilesystemBackend> filesystemBackend)
      : BlobCacheBackend(runtime), inmemoryBackend(std::move(inmemoryBackend)),
        filesystemBackend(std::move(filesystemBackend)) {}

  ErrorOrSuccess insertImpl(StringRef keyHash, BufferRef obj) override {
    // We only need to insert into the filesystem backend. When looking up a
    // result, that's when we'll populate the in-memory backend.
    return filesystemBackend->insertImpl(keyHash, std::move(obj));
  }

  ErrorOr<bool> containsImpl(StringRef keyHash) const override {
    auto containsOr = inmemoryBackend->containsImpl(keyHash);
    if (containsOr.isError() || *containsOr)
      return containsOr;
    return filesystemBackend->containsImpl(keyHash);
  }

  ErrorOr<std::optional<BufferRef>>
  findImpl(StringRef keyHash,
           std::optional<WriteableBufferRef> buf) const override {
    auto bufCopy = buf ? buf->copy() : std::optional<WriteableBufferRef>();
    auto result = inmemoryBackend->findImpl(keyHash, std::move(bufCopy));
    if (result.isError() || *result)
      return result;

    // If we didn't find it in the in-memory backend, try the filesystem
    // backend.
    result = filesystemBackend->findImpl(keyHash, std::move(buf));
    if (result.isError() || !*result)
      return result;

    // If we found it in the filesystem backend, insert it into the in-memory
    // backend.
    if (auto err = inmemoryBackend->insertImpl(keyHash, (*result)->copy()))
      return err.takeError();
    return result;
  }

  /// The in-memory backend used to store filesystem references.
  RCRef<InMemoryBackend> inmemoryBackend;

  /// The filsystem backend.
  RCRef<FilesystemBackend> filesystemBackend;
};
} // namespace

//===----------------------------------------------------------------------===//
// DylibBlobCacheBackend
//===----------------------------------------------------------------------===//

namespace {
/// This stub loads a shared library that implements a BlobCacheBackend
/// using the DylibBlobCacheBackend interface, and delegates backend calls
/// to this implementation.
struct DylibBackendStub : public BlobCacheBackend {

  static ErrorOr<RCRef<BlobCacheBackend>>
  create(LLCL::Runtime &runtime, StringRef libPath,
         const DylibBackendConfig *config) {
    auto backendStub = RCRef<DylibBackendStub>::create(runtime);
    if (auto err = backendStub->load(runtime, libPath, config))
      return err.takeError();
    return backendStub;
  }

  ~DylibBackendStub() override {
    // Release reference to ensure the backend is deleted before closing the
    // library.
    backend.reset();
    llvm::sys::DynamicLibrary::closeLibrary(dylib);
  }

  AsyncValueRef<Chain> insert(BufferRef keyHash, BufferRef obj,
                              std::optional<EncodedLocation> loc) override {
    return backend->insert(std::move(keyHash), std::move(obj), std::move(loc));
  }

  AsyncValueRef<bool> contains(BufferRef keyHash,
                               std::optional<EncodedLocation> loc) override {
    return backend->contains(std::move(keyHash), std::move(loc));
  }

  AsyncValueRef<std::optional<BufferRef>>
  find(BufferRef keyHash, std::optional<WriteableBufferRef> buf,
       std::optional<EncodedLocation> loc) override {
    return backend->find(std::move(keyHash), std::move(buf), std::move(loc));
  }

private:
  /// So RCRef can access private constructor.
  friend class RCRef<DylibBackendStub>;

  explicit DylibBackendStub(LLCL::Runtime &runtime)
      : BlobCacheBackend(runtime) {}

  ErrorOrSuccess load(LLCL::Runtime &runtime, StringRef libPath,
                      const DylibBackendConfig *config) {
    std::string errorMsg;
    dylib =
        llvm::sys::DynamicLibrary::getLibrary(libPath.str().c_str(), &errorMsg);
    if (!dylib.isValid())
      return Error("Failed to load library " + libPath + ": " + errorMsg);

    using allocType = DylibBlobCacheBackend *(*)(LLCL::Runtime * runtime);
    auto allocFunc = reinterpret_cast<allocType>(
        dylib.getAddressOfSymbol("M_CAS_allocateBackend"));
    if (!allocFunc) {
      llvm::sys::DynamicLibrary::closeLibrary(dylib);
      return Error("M_CAS_allocateBackend symbol not found\n");
    }
    backend = RCRef<DylibBlobCacheBackend>::take(allocFunc(&runtime));
    return backend->setConfig(config);
  }

  /// The dynamic library handle.
  llvm::sys::DynamicLibrary dylib;
  /// The stub delegates all cache-related operations to this backend.
  RCRef<DylibBlobCacheBackend> backend;
};
} // namespace

ErrorOr<RCRef<BlobCacheBackend>>
M::Cache::getS3Backend(LLCL::Runtime &runtime, const S3BackendConfig &config) {
#if defined(__linux__)
  constexpr llvm::StringLiteral libPath = "libblobcache_s3.so";
#elif defined(__APPLE__)
  constexpr llvm::StringLiteral libPath = "libblobcache_s3.dylib";
#elif defined(_WIN32)
  constexpr llvm::StringLiteral libPath = "blobcache_s3.dll";
#endif
  return DylibBackendStub::create(runtime, libPath, &config);
}

ErrorOr<RCRef<BlobCacheBackend>>
M::Cache::getLocalDefaultBackendChain(LLCL::Runtime &runtime,
                                      const std::filesystem::path &cacheDir,
                                      std::string version) {
  auto backend = RCRef<InMemoryBackend>::create(runtime);

  auto filesystemBackend =
      getVersionedFilesystemBackend(runtime, cacheDir, std::move(version));
  if (failed(filesystemBackend))
    return filesystemBackend.takeError();

  // Wrap the filesystem backend in an in-memory caching backend. This ensures
  // we only store mmap'd data in memory.
  return RCRef<FileSystemBackedInMemoryBackend>::create(
      runtime, backend.copy(), filesystemBackend->copy());
}

ErrorOr<RCRef<BlobCacheBackend>>
M::Cache::getDefaultBackendChain(LLCL::Runtime &runtime, const URI &uri,
                                 std::string version) {
  StringRef scheme = uri.getScheme();
  if (scheme == "file")
    return getLocalDefaultBackendChain(runtime, uri.getPath().str(), version);

  // If no version is specified, use the default version.
  if (version.empty())
    version = getModularVersionString();

  if (scheme == "s3") {
    StringRef path = uri.getPath();
    S3BackendConfig config(uri.getAuthority().str(),
                           path.str() + "/" + version);
    auto backendOr = getS3Backend(runtime, config);
    if (backendOr.isError())
      return backendOr.takeError();

    // Get a default local backend chain and add the s3 backend to the end.
    path.consume_front("/"); // Convert the path component to relative.
    auto localChainOr =
        getLocalDefaultBackendChain(runtime, path.str(), version);
    if (localChainOr.isError())
      return localChainOr.takeError();
    (*localChainOr)->appendDelegate(std::move(*backendOr));
    return localChainOr;
  }

  return Error("Can't build BlobCache backend chain with unknown URI scheme: " +
               scheme);
}
