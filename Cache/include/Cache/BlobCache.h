//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef CACHE_BLOBCACHE_H
#define CACHE_BLOBCACHE_H

#include "AsyncRT/Runtime/Allocator.h"
#include "AsyncRT/Runtime/AsyncValueRef.h"
#include "AsyncRT/Runtime/Runtime.h"
#include "Support/Buffer.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/STLExtras.h"
#include "Support/URI.h"
#include "llvm/Support/MemoryBuffer.h"
#include <filesystem>
#include <string_view>

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
  virtual ~BlobCacheBackend() {}

  /// Store the object `obj` with hash `keyHash`. This is expected to take
  /// ownership of the data in `obj` on success. Subclasses are expected to
  /// overwrite the current contents on a collision.
  LLCL::AsyncValueRef<LLCL::Chain>
  insert(LLCL::Runtime &runtime, BufferRef keyHash, BufferRef obj,
         std::optional<EncodedLocation> loc = std::nullopt);

  /// May be overwritten to provide an asynchronous insert.
  virtual LLCL::AsyncValueRef<LLCL::Chain>
  insertImpl(LLCL::Runtime &runtime, BufferRef keyHash, BufferRef obj,
             std::optional<EncodedLocation> loc = std::nullopt);

  /// Check if an item with key hash `keyHash` exists in this backend or in any
  /// of the delegates.
  LLCL::AsyncValueRef<bool>
  contains(LLCL::Runtime &runtime, BufferRef keyHash,
           std::optional<EncodedLocation> loc = std::nullopt);

  /// May be overwritten to provide an asynchronous contains.
  virtual LLCL::AsyncValueRef<bool>
  containsImpl(LLCL::Runtime &runtime, BufferRef keyHash,
               std::optional<EncodedLocation> loc = std::nullopt);

  /// Get the item with key hash `keyHash` from this backend or any of its
  /// delegates.
  LLCL::AsyncValueRef<std::optional<BufferRef>>
  find(LLCL::Runtime &runtime, BufferRef keyHash,
       std::optional<EncodedLocation> loc = std::nullopt);

  /// May be overwritten to provide an asynchronous find.
  virtual LLCL::AsyncValueRef<std::optional<BufferRef>>
  findImpl(LLCL::Runtime &runtime, BufferRef keyHash,
           std::optional<EncodedLocation> loc = std::nullopt);

  /// Subclasses that don't override insert should use this to provide the
  /// implementation of actually storing an item.
  ErrorOrSuccess insertSync(StringRef keyHash, BufferRef obj);

  /// Must be overwritten to provide synchronous insert.
  virtual ErrorOrSuccess insertSyncImpl(StringRef keyHash, BufferRef obj) = 0;

  /// Subclasses that don't override contains should use this to provide the
  /// implementation of checking if an item exists.
  ErrorOr<bool> containsSync(StringRef keyHash);

  /// Must be overwritten to provide synchronous contains.
  virtual ErrorOr<bool> containsSyncImpl(StringRef keyHash) = 0;

  /// Subclasses that don't override find should use this to provide the
  /// implementation of getting an item from storage.
  ErrorOr<std::optional<BufferRef>> findSync(StringRef keyHash);

  /// Must be overwritten to provide a synchronous find.
  virtual ErrorOr<std::optional<BufferRef>> findSyncImpl(StringRef keyHash) = 0;

  /// Add delegate to the end of the backend chain.
  virtual void appendDelegate(RCRef<BlobCacheBackend> d);

private:
  /// The next backend in the list. The public APIs handle nullptr here
  /// correctly, and the protected APIs (for the subclasses) should ignore the
  /// presence of this delegate entirely.
  RCRef<BlobCacheBackend> delegate;
};

class DylibBackendConfig {
public:
  enum ConfigKind {};

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
      : backendList(std::move(backendList)) {}

  using KeyTy = typename KeyInfo::KeyTy;

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
  insert(LLCL::Runtime &runtime, KeyTy key, BufferRef obj,
         std::optional<EncodedLocation> loc = std::nullopt) {
    std::string keyHash = KeyInfo::hashKey(std::forward<KeyTy>(key));
    LLCL::AsyncValueRef<LLCL::Chain> insertAsync =
        backendList->insert(runtime, Buffer::get(keyHash), std::move(obj));

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
  ErrorOr<std::string> insertSync(KeyTy key, BufferRef obj) {
    std::string keyHash = KeyInfo::hashKey(std::forward<KeyTy>(key));
    auto errOr = backendList->insertSync(keyHash, std::move(obj));
    if (errOr.isError())
      return errOr.takeError();
    return keyHash;
  }

  /// Check if any of the provided backends have the item.
  LLCL::AsyncValueRef<bool>
  contains(LLCL::Runtime &runtime, KeyTy key,
           std::optional<EncodedLocation> loc = std::nullopt) const {
    auto hash = Buffer::get(KeyInfo::hashKey(std::forward<KeyTy>(key)));
    return backendList->contains(runtime, std::move(hash), std::move(loc));
  }
  ErrorOr<bool> containsSync(KeyTy key) const {
    auto hash = KeyInfo::hashKey(std::forward<KeyTy>(key));
    return backendList->containsSync(hash);
  }

  /// Get the item from any of the provided backends.
  LLCL::AsyncValueRef<std::optional<BufferRef>>
  find(LLCL::Runtime &runtime, KeyTy key,
       std::optional<EncodedLocation> loc = std::nullopt) const {
    auto hash = Buffer::get(KeyInfo::hashKey(std::forward<KeyTy>(key)));
    return backendList->find(runtime, std::move(hash), std::move(loc));
  }
  ErrorOr<std::optional<BufferRef>> findSync(KeyTy key) const {
    auto hash = KeyInfo::hashKey(std::forward<KeyTy>(key));
    return backendList->findSync(hash);
  }

private:
  RCRef<BlobCacheBackend> backendList;
};

/// Returns an in-memory implementation of the BlobCacheBackend.
RCRef<BlobCacheBackend> getInMemoryBackend();

/// Returns a filesystem-based implementation of the BlobCacheBackend. If the
/// base path is not specified, then the backend will use the CWD. The cache
/// reads and writes to the filesystem by default, but if `readOnly` is
/// specified, only reads are performed.
RCRef<BlobCacheBackend>
getFilesystemBackend(const std::filesystem::path &basePath = "",
                     bool readOnly = false);

/// Returns a filesystem-based implementation of the BlobCacheBackend. The
/// `cacheDir` is used to derive a path for use by the filesystem backend. The
/// `version` specifies the version string of the cache, defaults to
/// MODULAR_VERSION_STRING if the provided version is empty.
ErrorOr<RCRef<BlobCacheBackend>>
getFilesystemBackend(const std::filesystem::path &cacheDir,
                     std::string_view version);

/// Returns a chain of pre-setup backends that represent the default chain,
/// inMemory->filesystem. The `cacheDir` is used to derive a path for use
/// by the filesystem backend. The `version` specifies the version string of the
/// cache, defaults to MODULAR_VERSION_STRING if the provided version is empty.
ErrorOr<RCRef<BlobCacheBackend>>
getLocalDefaultBackendChain(const std::filesystem::path &cacheDir = "",
                            std::string_view version = "");

ErrorOr<RCRef<BlobCacheBackend>>
getDefaultBackendChain(const URI &cacheUri, std::string_view version = "");

} // namespace M::Cache

#endif // CACHE_BLOBCACHE_H
