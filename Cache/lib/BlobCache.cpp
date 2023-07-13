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
#include "Support/FileSystemExtras.h"
#include "Support/HMAC.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Base64.h"
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
BlobCacheBackend::find(BufferRef keyHash, std::optional<WriteableBufferRef> buf,
                       std::optional<EncodedLocation> loc) {
  auto result = AsyncValueRef<std::optional<BufferRef>>::allocate(runtime);
  addTask(runtime, [thisRef = copyRCRef(this), keyHash = keyHash.copy(),
                    result = result.copy(), buf = std::move(buf),
                    loc = std::move(loc)]() mutable {
    // Find it at this level.
    ErrorOr<std::optional<BufferRef>> bufOr = thisRef->findImpl(
        keyHash->getBuffer(),
        (buf ? std::optional<WriteableBufferRef>(buf->copy()) : std::nullopt));
    if (bufOr.isError()) {
      return std::move(result).setToError(
          getError(std::move(loc), bufOr.takeError()));
    }

    // If we had it, return, and we're done.
    if (bufOr->has_value())
      return std::move(result).emplace(std::move(*bufOr));

    // If we don't have it, try with delegate.
    return thisRef->delegateFind(std::move(result), std::move(keyHash),
                                 std::move(buf), std::move(loc));
  });

  return result;
}

void BlobCacheBackend::delegateFind(
    AsyncValueRef<std::optional<BufferRef>> result, BufferRef keyHash,
    std::optional<WriteableBufferRef> buf, std::optional<EncodedLocation> loc) {
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
      (buf ? std::optional<WriteableBufferRef>(buf->copy()) : std::nullopt),
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

AsyncValueRef<Chain>
BlobCacheBackend::clear(std::optional<EncodedLocation> loc) {
  auto result = AsyncValueRef<Chain>::allocate(runtime);
  addTask(runtime, [thisRef = copyRCRef(this), result = result.copy(),
                    loc = std::move(loc)]() mutable {
    if (auto err = thisRef->clearImpl()) {
      return std::move(result).setToError(
          getError(std::move(loc), err.takeError()));
    }

    return thisRef->delegateClear(std::move(result), std::move(loc));
  });
  return result;
}

void BlobCacheBackend::delegateClear(AsyncValueRef<Chain> result,
                                     std::optional<EncodedLocation> loc) {
  if (!delegate)
    return std::move(result).emplace();

  auto clear = delegate->clear(std::move(loc));
  std::move(clear).andThenSync(
      [result = std::move(result)](AsyncValueRef<Chain> &&clear) mutable {
        if (clear.isError())
          return std::move(result).setToError(clear.takeDiagnostic());

        return std::move(result).emplace();
      });
}

void BlobCacheBackend::appendDelegate(LLCL::RCRef<BlobCacheBackend> d) {
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
           std::optional<WriteableBufferRef> buf) const override {
    std::shared_lock<std::shared_mutex> lock(mutex);
    auto found = cache.find(keyHash);
    if (found == cache.end())
      return std::nullopt;
    // No buffer provided, give back a ref to the buffer we have.
    if (!buf)
      return found->second.copy();

    // If we were passed in a buffer...
    Buffer &foundBuf = *found->second;
    // If the buffer already contains the data, don't bother doing anything.
    if ((*buf)->getBufferStart() == foundBuf.getBufferStart())
      return found->second.copy();

    if ((*buf)->getBufferSize() < foundBuf.getBufferSize())
      return Error("Buffer passed to CAS (size " +
                   Twine((*buf)->getBufferSize()) +
                   ") cannot accommodate found object (size " +
                   Twine(foundBuf.getBufferSize()) + ")");

    // Write the contents of the buffer we found to offset 0.
    (*buf)->pwrite(foundBuf.getBufferStart(), foundBuf.getBufferSize(), 0);
    // And return a ref to *that* buffer.
    return buf->copy();
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
/// Provides a filesystem-backed backend that stores the buffers in binary
/// files on disk.
struct FilesystemBackend : public BlobCacheBackend {
  explicit FilesystemBackend(LLCL::Runtime &runtime,
                             const std::filesystem::path &basePath)
      : BlobCacheBackend(runtime), basePath(basePath.string()) {}

  ErrorOrSuccess insertImpl(StringRef keyHash, BufferRef obj) override {
    // Check if we already have the object - if we do, then don't bother writing
    // it again.
    ErrorOr<bool> containsOr = containsImpl(keyHash);
    if (!containsOr.isError() && *containsOr)
      return success();

    // Get the absolute path and create any directories we need to create.
    std::filesystem::path filePath = getAbsolutePathForKey(keyHash);
    std::error_code dirErr;
    std::filesystem::create_directories(filePath.parent_path(), dirErr);
    if (dirErr)
      return Error(dirErr.message());

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
    if (auto err = writeFileAtomically(filePath, writeContent); err.isError())
      return err.takeError();

    return success();
  }

  ErrorOr<bool> containsImpl(StringRef keyHash) const override {
    auto abs = getAbsolutePathForKey(keyHash);
    return std::filesystem::exists(abs) && !std::filesystem::is_directory(abs);
  }

  ErrorOr<std::optional<BufferRef>>
  findImpl(StringRef keyHash,
           std::optional<WriteableBufferRef> buf) const override {
    // Get the file path and open it.
    std::filesystem::path filePath = getAbsolutePathForKey(keyHash);
    // No such file, return nullopt (not error).
    if (!std::filesystem::exists(filePath))
      return std::nullopt;

    // This callback does all the file reading/etc. we care to do while we hold
    // the lock on the file - that's why it's such a large inline lambda.
    std::optional<Error> error = std::nullopt;
    BufferRef out;
    auto doRead = [&error, buf = std::move(buf),
                   &out](const std::filesystem::path &path) mutable {
      auto bufOr = Buffer::getFile(path);
      // If the file doesn't exist, or it's empty, return an error.
      if (bufOr.isError()) {
        error = bufOr.takeError();
        return;
      }

      BufferRef buffer = std::move(*bufOr);
      if (buffer->getBufferSize() == 0) {
        error =
            Error("file '" + Twine(path.string()) + "' exists, but is empty");
        return;
      }

      StringRef contentsAndHMAC = buffer->getBuffer();

      // Get a StringRef of the contents without the HMAC.
      StringRef contents = contentsAndHMAC.drop_back(blake3Bytes);
      BLAKE3Hash computedHMAC = hmacBLAKE3(contents, kIntegrityKey);
      StringRef storedHMAC = contentsAndHMAC.take_back(blake3Bytes);

      // Check the computed hmac against the one in the file.
      if (memcmp(computedHMAC.data(), storedHMAC.data(), blake3Bytes) != 0) {
        error = Error("corrupted file: stored hash and computed hash did not "
                      "match for file '" +
                      Twine(path.string()) + "'");
        return;
      }

      // Now that we've verified the integrity of the file, return a memory
      // buffer that holds just the contents.
      bufOr = Buffer::getFile(path, contents.size(),
                              /*offset=*/0);
      if (failed(bufOr)) {
        error = bufOr.takeError();
        return;
      }

      // No buffer provided, return the mapped thing directly.
      if (!buf) {
        out = BufferRef::take(bufOr->release());
        return;
      }

      if ((*buf)->getBufferSize() < (*bufOr)->getBufferSize())
        error = Error("Buffer passed to CAS (size " +
                      Twine((*buf)->getBufferSize()) +
                      ") cannot accommodate found object (size " +
                      Twine((*bufOr)->getBufferSize()) + ")");

      // Copy the file data into the buffer that was provided to us.
      (*buf)->pwrite((*bufOr)->getBufferStart(), (*bufOr)->getBufferSize(), 0);
      // Take a reference to the provided buffer and return it.
      out = BufferRef::copy((*buf).getPointer());
    };

    // If there was an error reading the file, return that.
    if (auto err = readFileAtomically(filePath, doRead))
      return err.takeError();

    // If there was an error in our read callback, return that.
    if (error)
      return std::move(*error);

    // Otherwise, return the BufferRef we created.
    return std::move(out);
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

//===----------------------------------------------------------------------===//
// DylibBlobCacheBackend
//===----------------------------------------------------------------------===//

namespace {
/// This stub loads a shared library that implements a BlobCacheBackend
/// using the DylibBlobCacheBackend interface, and delegates backend calls
/// to this implementation.
struct DylibBackendStub : public BlobCacheBackend {

  static ErrorOr<LLCL::RCRef<BlobCacheBackend>>
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

  AsyncValueRef<Chain> clear(std::optional<EncodedLocation> loc) override {
    return backend->clear(std::move(loc));
  }

private:
  /// So RCRef can access private constructor.
  friend class LLCL::RCRef<DylibBackendStub>;

  explicit DylibBackendStub(LLCL::Runtime &runtime)
      : BlobCacheBackend(runtime) {}

  ErrorOrSuccess load(LLCL::Runtime &runtime, StringRef libPath,
                      const DylibBackendConfig *config) {
    std::string errorMsg;
    dylib =
        llvm::sys::DynamicLibrary::getLibrary(libPath.str().c_str(), &errorMsg);
    if (!dylib.isValid()) {
      return Error("Failed to load library " + libPath + ": " + errorMsg);
    }

    using allocType = DylibBlobCacheBackend *(*)(LLCL::Runtime * runtime);
    auto allocFunc = reinterpret_cast<allocType>(
        dylib.getAddressOfSymbol("M_CAS_allocateBackend"));
    if (!allocFunc) {
      llvm::sys::DynamicLibrary::closeLibrary(dylib);
      return Error("M_CAS_allocateBackend symbol not found\n");
    }
    backend = LLCL::RCRef<DylibBlobCacheBackend>::take(allocFunc(&runtime));
    return backend->setConfig(config);
  }

  /// The dynamic library handle.
  llvm::sys::DynamicLibrary dylib;
  /// The stub delegates all cache-related operations to this backend.
  LLCL::RCRef<DylibBlobCacheBackend> backend;
};
} // namespace

ErrorOr<LLCL::RCRef<BlobCacheBackend>>
M::Cache::getS3Backend(LLCL::Runtime &runtime, const S3BackendConfig &config) {
#if defined(__linux__)
  constexpr llvm::StringLiteral libPath = "libblobcache_s3.so";
#elif defined(__APPLE__)
  constexpr llvm::StringLiteral libPath = "libblobcache_s3.dylib";
#elif defined(WIN32)
  constexpr llvm::StringLiteral libPath = "blobcache_s3.dll";
#endif
  return DylibBackendStub::create(runtime, libPath, &config);
}

ErrorOr<LLCL::RCRef<BlobCacheBackend>>
M::Cache::getLocalDefaultBackendChain(LLCL::Runtime &runtime,
                                      const std::filesystem::path &cacheDir,
                                      std::string version) {
  auto backend = getInMemoryBackend(runtime);

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
#ifdef _WIN32
    } else if (auto path = findDirInEnvPath(cacheDir.string(), "PATH", ';')) {
#else
    } else if (auto path = findDirInEnvPath(cacheDir.string())) {
#endif
      base = std::filesystem::absolute(*path, ec);
      if (ec)
        return Error("failed to get absolute path to directory specified by " +
                     *path + ec.message());
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
          (std::filesystem::canonical(dirEntry.path().parent_path()) ==
           std::filesystem::canonical(base)) &&
          (dirEntry.path().filename() != version)) {
        std::filesystem::remove_all(dirEntry, ec);
      }
    }
  }

  base = base / version;

  backend->appendDelegate(getFilesystemBackend(runtime, base));
  return backend;
}

ErrorOr<LLCL::RCRef<BlobCacheBackend>>
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
