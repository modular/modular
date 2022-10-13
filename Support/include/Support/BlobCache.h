//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_BLOBCACHE_H
#define SUPPORT_BLOBCACHE_H

#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/Support/MemoryBufferRef.h"
#include <filesystem>

namespace M {
/// This class is the backend interface for a BlobCache. The backend contains a
/// pointer to its delegate, which is meant to be used as an option if this
/// backend has a cache miss. This means that the backends should be ordered on
/// priority - i.e. have an in-memory backend delegate to a remote backend, not
/// the other way around!
///
/// Conceptually, the backends form a linked-list that's sorted in priority
/// order that the BlobCache below will use to find an item.
class BlobCacheBackend {
public:
  virtual ~BlobCacheBackend() = default;

  /// Store the object `obj` with hash `keyHash`. This is expected to take
  /// ownership of the data in `obj` on success. Subclasses are expected to
  /// overwrite the current contents on a collision.
  ErrorOrSuccess insert(StringRef keyHash, llvm::MemoryBufferRef obj);

  /// Check if an item with key hash `keyHash` exists in this backend or in any
  /// of the delegates.
  bool contains(StringRef keyHash);

  /// Get the item with key hash `keyHash` from this backend or any of its
  /// delegates.
  ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> find(StringRef keyHash);

  /// Overwrite the current delegate.
  void setDelegate(std::unique_ptr<BlobCacheBackend> &&d) {
    delegate = std::move(d);
  }

protected:
  /// Subclasses should use this to provide the implementation of actually
  /// storing an item.
  virtual ErrorOrSuccess insertImpl(StringRef keyHash,
                                    llvm::MemoryBufferRef obj) = 0;
  /// Subclasses should use this to provide the implementation of checking if an
  /// item exists.
  virtual bool containsImpl(StringRef keyHash) const = 0;
  /// Subclasses should use this to provide the implementation of getting an
  /// item from storage.
  virtual ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
  findImpl(StringRef keyHash) const = 0;

private:
  /// The next backend in the list. The public APIs handle nullptr here
  /// correctly, and the protected APIs (for the subclasses) should ignore the
  /// presence of this delegate entirely.
  std::unique_ptr<BlobCacheBackend> delegate = nullptr;
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
class BlobCache {
public:
  explicit BlobCache(std::unique_ptr<BlobCacheBackend> &&backendList)
      : backendList(std::move(backendList)) {}

  using KeyTy = typename KeyInfo::KeyTy;

  /// Store an item in the provided backends. On a collision, the backends are
  /// expected to overwrite the existing contents, so it is incumbent on the
  /// user to use a strong hash function! Returns the cache key on success -
  /// this can be used for speeding up future hash computations or simply
  /// discarded.
  ErrorOr<std::string> insert(KeyTy key, llvm::MemoryBufferRef obj) {
    std::string keyHash = KeyInfo::hashKey(key);
    if (auto err = backendList->insert(keyHash, obj))
      return err.takeError();

    return keyHash;
  }

  /// Check if any of the provided backends have the item.
  bool contains(KeyTy key) const {
    return backendList->contains(KeyInfo::hashKey(key));
  }

  /// Get the item from any of the provided backends.
  ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> find(KeyTy key) {
    return backendList->find(KeyInfo::hashKey(key));
  }

private:
  std::unique_ptr<BlobCacheBackend> backendList;
};

/// Returns an in-memory implementation of the BlobCacheBackend.
std::unique_ptr<BlobCacheBackend> getInMemoryBackend();

/// Returns a filesystem-based implementation of the BlobCacheBackend. If the
/// base path is not specified, then the backend will use the CWD.
std::unique_ptr<BlobCacheBackend>
getFilesystemBackend(const std::filesystem::path &basePath = "");

/// Returns a chain of pre-setup backends that represent the default chain,
/// inMemory->filesystem. The `basePath` is passed to getFilesystemBackend
/// directly.
std::unique_ptr<BlobCacheBackend>
getDefaultBackendChain(const std::filesystem::path &basePath = "");
} // namespace M

#endif // SUPPORT_BLOBCACHE_H
