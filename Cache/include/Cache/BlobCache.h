//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef CACHE_BLOBCACHE_H
#define CACHE_BLOBCACHE_H

#include "LLCL/Runtime/Allocator.h"
#include "LLCL/Runtime/AsyncValueRef.h"
#include "LLCL/Runtime/Runtime.h"
#include "Support/Buffer.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/STLExtras.h"
#include "Support/URI.h"
#include "llvm/Support/MemoryBuffer.h"

#include <filesystem>

namespace M::Cache {
/// This class is the backend interface for a BlobCache. The backend contains a
/// pointer to its delegate, which is meant to be used as an option if this
/// backend has a cache miss. This means that the backends should be ordered on
/// priority - i.e. have an in-memory backend delegate to a remote backend, not
/// the other way around!
///
/// Conceptually, the backends form a linked-list that's sorted in priority
/// order that the BlobCache below will use to find an item.
class BlobCacheBackend : public ReferenceCounted<BlobCacheBackend> {
public:
  /// Construct a BlobCacheBackend from an LLCL runtime.
  BlobCacheBackend(LLCL::Runtime &runtime) : runtime(runtime) {}
  virtual ~BlobCacheBackend() {}

  /// Return a reference to the LLCL runtime the backend was created with.
  LLCL::Runtime &getRuntime() { return runtime; }

  /// Store the object `obj` with hash `keyHash`. This is expected to take
  /// ownership of the data in `obj` on success. Subclasses are expected to
  /// overwrite the current contents on a collision.
  virtual LLCL::AsyncValueRef<LLCL::Chain>
  insert(BufferRef keyHash, BufferRef obj,
         std::optional<EncodedLocation> loc = std::nullopt);

  /// Check if an item with key hash `keyHash` exists in this backend or in any
  /// of the delegates.
  virtual LLCL::AsyncValueRef<bool>
  contains(BufferRef keyHash,
           std::optional<EncodedLocation> loc = std::nullopt);

  /// Get the item with key hash `keyHash` from this backend or any of its
  /// delegates. If `backingBuf` is provided, write into that (and return a
  /// read-only reference to it if found).
  virtual LLCL::AsyncValueRef<std::optional<BufferRef>>
  find(BufferRef keyHash,
       std::optional<WriteableBufferRef> backingBuf = std::nullopt,
       std::optional<EncodedLocation> loc = std::nullopt);

  /// Get all the items in `map` and emplace the AsyncValueRefs contained
  /// therein. The caller must ensure that the AsyncValueRef objects are
  /// allocated, this method will emplace them. This simply provides a default
  /// implementation - backends are expected to override this method if they
  /// wish to handle group finds differently than multiple individual finds. If
  /// `backingBuf` is provided, then we can return offsets to it.
  virtual void
  find(llvm::StringMap<LLCL::AsyncValueRef<std::optional<BufferRef>>> &map,
       std::optional<WriteableBufferRef> backingBuf = std::nullopt,
       std::optional<EncodedLocation> loc = std::nullopt);

  /// Clear out this backend and its delegates.
  virtual LLCL::AsyncValueRef<LLCL::Chain>
  clear(std::optional<EncodedLocation> loc = std::nullopt);

  /// Add delegate to the end of the backend chain.
  void appendDelegate(RCRef<BlobCacheBackend> d);

protected:
  /// NOTE: The asynchrony of the cache backend is handled by the
  /// BlobCacheBackend class so the *Impl functions can more or less ignore it.

  /// Subclasses that don't override insert should use this to provide the
  /// implementation of actually storing an item.
  virtual ErrorOrSuccess insertImpl(StringRef keyHash, BufferRef obj) {
    return Error("insertImpl not implemented");
  }
  /// Subclasses that don't override contains should use this to provide the
  /// implementation of checking if an item exists.
  virtual ErrorOr<bool> containsImpl(StringRef keyHash) const {
    return Error("containsImpl not implemented");
  }
  /// Subclasses that don't override find should use this to provide the
  /// implementation of getting an item from storage. If `backingBuf` is
  /// provided, write directly into `backingBuf`, returning std::nullopt if the
  /// item isn't found as usual.
  virtual ErrorOr<std::optional<BufferRef>>
  findImpl(StringRef keyHash,
           std::optional<WriteableBufferRef> backingBuf = std::nullopt) const {
    return Error("findImpl not implemented");
  }
  /// Subclasses should use this to provide the implementation of clearing the
  /// cache. Subclasses may choose not to provide this, for example, a cloud
  /// storage backend may not wish to actually clear all its storage. Backends
  /// are advised to kick off a clear operation asynchronously.
  virtual ErrorOrSuccess clearImpl() { return success(); }

  /// Use delegate to insert an item and set the status in the provided
  /// AsyncValue. This is called by the default insert, and can optionally be
  /// used by subclasses that override insert.
  void delegateInsert(LLCL::AsyncValueRef<LLCL::Chain> result,
                      BufferRef keyHash, BufferRef obj,
                      std::optional<LLCL::EncodedLocation> loc = std::nullopt);

  /// Use delegate to check if an item exists, and set the status in the
  /// provided AsyncValue. This is called by the default contains, and can
  /// optionally be used by subclasses that override contains.
  void
  delegateContains(LLCL::AsyncValueRef<bool> result, BufferRef keyHash,
                   std::optional<LLCL::EncodedLocation> loc = std::nullopt);

  /// Use delegate to find an item and set the result in the provided
  /// AsyncValue. This is called by the default find, and can optionally be used
  /// by subclasses that override find.
  void delegateFind(LLCL::AsyncValueRef<std::optional<BufferRef>> result,
                    BufferRef keyHash,
                    std::optional<WriteableBufferRef> backingBuf,
                    std::optional<EncodedLocation> loc = std::nullopt);

  /// Clear delegate cache. This is called by the default clear, and can
  /// optionally be used by subclasses that override clear.
  void delegateClear(LLCL::AsyncValueRef<LLCL::Chain> result,
                     std::optional<LLCL::EncodedLocation> loc = std::nullopt);

  /// The LLCL runtime we should use for managing asynchrony.
  LLCL::Runtime &runtime;

private:
  /// The next backend in the list. The public APIs handle nullptr here
  /// correctly, and the protected APIs (for the subclasses) should ignore the
  /// presence of this delegate entirely.
  RCRef<BlobCacheBackend> delegate;
};

class DylibBackendConfig {
public:
  enum ConfigKind {
    kS3,
  };

  DylibBackendConfig(ConfigKind kind) : kind(kind) {}

  ConfigKind getKind() const { return kind; }

private:
  const ConfigKind kind;
};

/// This is the interface for backends that are implemented in a shared library
/// and opened with dlopen (or other OS equivalent). This is meant to be
/// generic.
class DylibBlobCacheBackend : public BlobCacheBackend {
public:
  DylibBlobCacheBackend(LLCL::Runtime &runtime) : BlobCacheBackend(runtime) {}

  virtual ~DylibBlobCacheBackend() = default;

  virtual ErrorOrSuccess setConfig(const DylibBackendConfig *config) = 0;
};

/// This is the thing that users will interact with. It holds onto the list of
/// backends and calls into them, but its primary responsibility is to hash the
/// keys passed in to normalize the way we try to access the storage backends.
/// This cache supports any key type that can be hashed as long as the hash
/// method is provided through the `KeyInfo` type.
///
/// \tparam KeyInfo A struct that describes how to handle a key. It should be of
/// the form:
///
///   struct KeyInfo {
///     using KeyTy = SomeT;
///     static std::string hashKey(KeyTy key);
///   }
///
template <typename KeyInfo>
class BlobCache : public ReferenceCounted<BlobCache<KeyInfo>> {
public:
  explicit BlobCache(RCRef<BlobCacheBackend> backendList)
      : runtime(backendList->getRuntime()),
        backendList(std::move(backendList)) {}

  using KeyTy = typename KeyInfo::KeyTy;

  LLCL::Runtime &getRuntime() { return runtime; }

  /// Simple method to get the hash of a key via the KeyInfo struct. This is
  /// useful if (for example) we already have the object in the cache.
  std::string getHash(KeyTy key) const {
    return KeyInfo::hashKey(std::forward<KeyTy>(key));
  }

  /// Store an item in the provided backends. On a collision, the backends are
  /// expected to overwrite the existing contents, so it is incumbent on the
  /// user to use a strong hash function! Returns the cache key on success -
  /// this can be used for speeding up future hash computations or simply
  /// discarded.
  LLCL::AsyncValueRef<std::string>
  insert(KeyTy key, BufferRef obj,
         std::optional<EncodedLocation> loc = std::nullopt) {
    std::string keyHash = KeyInfo::hashKey(std::forward<KeyTy>(key));
    LLCL::AsyncValueRef<LLCL::Chain> insertAsync =
        backendList->insert(Buffer::get(keyHash), std::move(obj));

    // Allocate a space for the output.
    auto out = LLCL::AsyncValueRef<std::string>::allocate(runtime);
    std::move(insertAsync)
        .andThenSync([keyHash = std::move(keyHash), out = out.copy()](
                         AsyncValueRef<LLCL::Chain> &&insertAsync) mutable {
          // If insertion failed, propagate the error. Otherwise, hand over the
          // key hash.
          if (insertAsync.isError())
            return std::move(out).setToError(insertAsync.takeDiagnostic());

          return std::move(out).emplace(keyHash);
        });

    return out;
  }

  /// Check if any of the provided backends have the item.
  LLCL::AsyncValueRef<bool>
  contains(KeyTy key, std::optional<EncodedLocation> loc = std::nullopt) const {
    auto hash = Buffer::get(KeyInfo::hashKey(std::forward<KeyTy>(key)));
    return backendList->contains(std::move(hash), std::move(loc));
  }

  /// Get the item from any of the provided backends.
  LLCL::AsyncValueRef<std::optional<BufferRef>>
  find(KeyTy key, std::optional<EncodedLocation> loc = std::nullopt) {
    auto hash = Buffer::get(KeyInfo::hashKey(std::forward<KeyTy>(key)));
    return backendList->find(std::move(hash), std::nullopt, std::move(loc));
  }

  /// Get the item from any of the provided backends, reading it directly into
  /// `buf`. Returns a read-only ref to the buffer that was passed in if the
  /// item is found, std::nullopt otherwise.
  LLCL::AsyncValueRef<std::optional<BufferRef>>
  find(KeyTy key, WriteableBufferRef backingBuf,
       std::optional<EncodedLocation> loc = std::nullopt) {
    auto hash = Buffer::get(KeyInfo::hashKey(std::forward<KeyTy>(key)));
    return backendList->find(std::move(hash), std::move(backingBuf),
                             std::move(loc));
  }

  /// Get the items from any of the provided backends. This will attempt to
  /// fetch all the items in the array `keys`. Returns a map from the key's hash
  /// to the returned buffer. Every key will be found in the returned map, but
  /// the value may resolve to std::nullopt if it's not found in the cache. In
  /// case of an error, the individual entries will be set to error.
  // TODO: Provide a version that can take a single writable buffer and return
  //       aliases to it.
  llvm::StringMap<LLCL::AsyncValueRef<std::optional<BufferRef>>>
  find(ArrayRef<KeyTy> keys,
       std::optional<EncodedLocation> loc = std::nullopt) {
    llvm::StringMap<LLCL::AsyncValueRef<std::optional<BufferRef>>> map;
    for (auto k : keys) {
      map[KeyInfo::hashKey(std::forward<KeyTy>(k))] =
          AsyncValueRef<std::optional<BufferRef>>::allocate(runtime);
    }

    // Do the find, and return the map.
    backendList->find(map, std::nullopt, std::move(loc));
    return map;
  }

  /// Get the items from any of the provided backends. This will attempt to
  /// fetch all the items in the array `keys`. Returns a map from the key's hash
  /// to the returned buffer. Every key will be found in the returned map, but
  /// the value may resolve to std::nullopt if it's not found in the cache. In
  /// case of an error, the individual entries will be set to error. All the
  /// returned BufferRefs will be offsets into `backingBuf`.
  llvm::StringMap<LLCL::AsyncValueRef<std::optional<BufferRef>>>
  find(ArrayRef<KeyTy> keys, WriteableBufferRef backingBuf,
       std::optional<EncodedLocation> loc = std::nullopt) {
    llvm::StringMap<LLCL::AsyncValueRef<std::optional<BufferRef>>> map;
    for (auto k : keys) {
      map[KeyInfo::hashKey(std::forward<KeyTy>(k))] =
          AsyncValueRef<std::optional<BufferRef>>::allocate(runtime);
    }

    // Do the find, and return the map.
    backendList->find(map, std::move(backingBuf), std::move(loc));
    return map;
  }

  LLCL::AsyncValueRef<LLCL::Chain>
  clear(std::optional<EncodedLocation> loc = std::nullopt) {
    return backendList->clear(std::move(loc));
  }

private:
  LLCL::Runtime &runtime;

  RCRef<BlobCacheBackend> backendList;
};

/// Returns an in-memory implementation of the BlobCacheBackend.
RCRef<BlobCacheBackend> getInMemoryBackend(LLCL::Runtime &runtime);

/// Returns a filesystem-based implementation of the BlobCacheBackend. If the
/// base path is not specified, then the backend will use the CWD. The cache
/// reads and writes to the filesystem by default, but if `readOnly` is
/// specified, only reads are performed.
RCRef<BlobCacheBackend>
getFilesystemBackend(LLCL::Runtime &runtime,
                     const std::filesystem::path &basePath = "",
                     bool readOnly = false);

class S3BackendConfig : public DylibBackendConfig {
public:
  S3BackendConfig(std::string bucket, std::string prefix,
                  size_t numIOThreads = 0)
      : DylibBackendConfig(ConfigKind::kS3), bucket(std::move(bucket)),
        prefix(std::move(prefix)), numIOThreads(numIOThreads) {}
  static bool classof(const DylibBackendConfig *config) {
    return config->getKind() == ConfigKind::kS3;
  }
  /// Bucket name.
  std::string bucket;
  /// Bucket region.
  std::string region;
  /// Prefix in S3 bucket for cache.
  std::string prefix;
  /// AWS thread pool size (number of threads to use for S3 IO). If 0,
  /// the S3 backend will decide (right now it will pick double the
  /// number of LLCL threads).
  size_t numIOThreads;
};

/// Returns a BlobCacheBackend that uses S3 for storage. This accepts the S3
/// config (which includes the bucket, region and prefix to use inside the
/// bucket for cached objects).
ErrorOr<RCRef<BlobCacheBackend>> getS3Backend(LLCL::Runtime &runtime,
                                              const S3BackendConfig &config);

/// Returns a chain of pre-setup backends that represent the default chain,
/// inMemory->filesystem. The `cacheDir` is used to derive a path for use
/// by the filesystem backend. The `version` specifies the version string of the
/// cache, defaults to MODULAR_VERSION_STRING if the provided version is empty.
ErrorOr<RCRef<BlobCacheBackend>>
getLocalDefaultBackendChain(LLCL::Runtime &runtime,
                            const std::filesystem::path &cacheDir = "",
                            std::string version = "");

ErrorOr<RCRef<BlobCacheBackend>>
getDefaultBackendChain(LLCL::Runtime &runtime, const URI &cacheUri,
                       std::string version = "");

/// Helper class to hold a BlobStore over KeyT and associated runtime.
/// If no existing runtime is available a default runtime is created.
/// The BlobStore will be instantiated using the 'default' backend chain
/// using the given cacheDir.
template <typename KeyT>
class RuntimeAndCache {
public:
  using CacheRef = RCRef<BlobCache<KeyT>>;

  /// Captures the cache directory and optional existing runtime. However
  /// the object is not valid until setup is called and returns success.
  RuntimeAndCache(std::string cacheDir = "",
                  Runtime *optExistingRuntime = nullptr)
      : cacheDir(std::move(cacheDir)), optExistingRuntime(optExistingRuntime) {}

  /// Set up the runtime and cache. The version string is passed directly to
  /// `getDefaultBackendChain` - it will commonly be KGEN_VERSION_STRING or
  /// similar.
  ErrorOrSuccess setup(std::string version = "") {
    assert(!cacheRef && "setup already called");
    auto uriOr = URI::parse(cacheDir);
    if (uriOr.isError())
      return uriOr.takeError();
    ownedRuntime = ConditionallyOwnedPointer<Runtime>::allocateIfNeeded(
        optExistingRuntime,
        LLCL::createLeakCheckAllocator(LLCL::createMallocAllocator()),
        LLCL::createSingleThreadWorkQueue());
    auto backendList =
        getDefaultBackendChain(*ownedRuntime, *uriOr, std::move(version));
    if (backendList.isError())
      return backendList.takeError();
    cacheRef = CacheRef::create(std::move(*backendList));
    return success();
  }

  bool isValid() const { return cacheRef.getPointer() != nullptr; }
  Runtime &getRuntime() {
    assert(isValid());
    return *ownedRuntime;
  }
  CacheRef getCacheRef() {
    assert(isValid());
    return cacheRef.copy();
  }
  BlobCache<KeyT> &getCache() {
    assert(isValid());
    return *cacheRef;
  }

private:
  std::string cacheDir;
  Runtime *optExistingRuntime;
  ConditionallyOwnedPointer<Runtime> ownedRuntime;
  CacheRef cacheRef;
};

} // namespace M::Cache

#endif // CACHE_BLOBCACHE_H
