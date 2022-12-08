//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef CACHE_BLOBCACHE_H
#define CACHE_BLOBCACHE_H

#include "Cache/Buffer.h"
#include "LLCL/Runtime/Allocator.h"
#include "LLCL/Runtime/AsyncValueRef.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/Support/MemoryBuffer.h"

#include <filesystem>

namespace M::Cache {
/// This type allows the BlobCache to differentiate between "an error
/// occurred" and "the object was not found in the cache". This is because
/// something not being in the cache isn't necessarily an error - that's a
/// policy decision we'd like to leave to clients.
struct CacheFindResult {
  /// Construct a CacheFindResult that indicates there's nothing in the cache.
  static CacheFindResult notInCache() { return {}; }

  /// Construct a CacheFindResult with a value. The CacheFindResult takes
  /// ownership of the value.
  static CacheFindResult value(BufferRef value) { return {std::move(value)}; }

  /// Construct a CacheFindResult with an error. The CacheFindResult takes
  /// ownership of the error.
  static CacheFindResult error(Error err) { return {std::move(err)}; }

  /// Returns true if this holds a value. Returns false if an error occurred
  /// OR if the requested key was not in the cache.
  bool hasValue() const { return !valueOr.isError() && valueOr->has_value(); }

  /// Returns true if an error occurred.
  bool isError() const { return valueOr.isError(); }

  /// Take the value held in this result. This object is in an undefined state
  /// after this returns.
  BufferRef takeValue() { return std::move(get()); }

  /// Get a BufferRef from the underlying memory buffer.
  const BufferRef &operator*() const { return get(); }

  /// Get the error string held by the underlying ErrorOr. The result
  /// maintains ownership of the string.
  const char *getError() const { return valueOr.getError(); }

  /// Take the error in the underlying ErrorOr. The result is in an undefined
  /// state after this returns.
  Error takeError() { return valueOr.takeError(); }

private:
  /// Construct the CacheFindResult with an error - this puts it in the error
  /// state.
  CacheFindResult(Error error) : valueOr(std::move(error)) {}
  /// Construct the CacheFindResult with a value - this puts it in the value
  /// state.
  CacheFindResult(BufferRef value) : valueOr(std::move(value)) {}
  /// Construct the CacheFindResult with nothing - this puts it in the "not in
  /// cache" state.
  CacheFindResult() : valueOr(llvm::None) {}

  /// Provide a safe getter. This is private because we want the user to
  /// explicitly take ownership of the value rather than leaving it sitting in
  /// this result object if they're going to return it or something.
  BufferRef &get() {
    assert(!valueOr.isError() && valueOr->has_value());
    return valueOr.get().value();
  }

  const BufferRef &get() const {
    return const_cast<CacheFindResult *>(this)->get();
  }

  /// This can be an error, "not in cache", or it can have a value. Error is
  /// indicated by having an error, while "not in cache" is indicated by
  /// llvm::None in the Optional, but no error in the ErrorOr. A value is
  /// indicated by having a value in the Optional *and* no error in the ErrorOr.
  ErrorOr<Optional<BufferRef>> valueOr;
};

/// This class is the backend interface for a BlobCache. The backend contains a
/// pointer to its delegate, which is meant to be used as an option if this
/// backend has a cache miss. This means that the backends should be ordered on
/// priority - i.e. have an in-memory backend delegate to a remote backend, not
/// the other way around!
///
/// Conceptually, the backends form a linked-list that's sorted in priority
/// order that the BlobCache below will use to find an item.
class BlobCacheBackend : public LLCL::ReferenceCounted<BlobCacheBackend> {
public:
  /// Construct a BlobCacheBackend from an LLCL runtime.
  BlobCacheBackend(LLCL::Runtime &runtime) : runtime(runtime) {
    // Register the types we use in the blob cache.
    LLCL::AsyncValue::registerTypes<ErrorOrSuccess, bool, CacheFindResult>();
  }
  virtual ~BlobCacheBackend() {}

  /// Return a reference to the LLCL runtime the backend was created with.
  LLCL::Runtime &getRuntime() { return runtime; }

  /// Store the object `obj` with hash `keyHash`. This is expected to take
  /// ownership of the data in `obj` on success. Subclasses are expected to
  /// overwrite the current contents on a collision.
  LLCL::AsyncValueRef<ErrorOrSuccess> insert(BufferRef keyHash, BufferRef obj);

  /// Check if an item with key hash `keyHash` exists in this backend or in any
  /// of the delegates.
  LLCL::AsyncValueRef<bool> contains(BufferRef keyHash);

  /// Get the item with key hash `keyHash` from this backend or any of its
  /// delegates.
  LLCL::AsyncValueRef<CacheFindResult> find(BufferRef keyHash);

  /// Clear out this backend and its delegates.
  LLCL::AsyncValueRef<ErrorOrSuccess> clear();

  /// Overwrite the current delegate.
  void setDelegate(LLCL::RCRef<BlobCacheBackend> d) { delegate = std::move(d); }

protected:
  /// NOTE: The asynchrony of the cache backend is handled by the
  /// BlobCacheBackend class so the *Impl functions can more or less ignore it.

  /// Subclasses should use this to provide the implementation of actually
  /// storing an item.
  virtual ErrorOrSuccess insertImpl(StringRef keyHash, BufferRef obj) = 0;
  /// Subclasses should use this to provide the implementation of checking if an
  /// item exists.
  virtual bool containsImpl(StringRef keyHash) const = 0;
  /// Subclasses should use this to provide the implementation of getting an
  /// item from storage.
  virtual CacheFindResult findImpl(StringRef keyHash) const = 0;
  /// Subclasses should use this to provide the implementation of clearing the
  /// cache. Subclasses may choose not to provide this, for example, a cloud
  /// storage backend may not wish to actually clear all its storage. Backends
  /// are advised to kick off a clear operation asynchronously.
  virtual ErrorOrSuccess clearImpl() { return success(); }

  /// Create a ready AsyncValueRef. This is just nice sugar to clean up the
  /// callsites when we return an AsyncValueRef that's ready to go, such as
  /// reporting an error.
  template <typename T>
  LLCL::AsyncValueRef<T> createReady(T &&val) {
    return LLCL::AsyncValueRef<T>::createReady(runtime,
                                               std::forward<T &&>(val));
  }

private:
  /// The LLCL runtime we should use for managing asynchrony.
  LLCL::Runtime &runtime;
  /// The next backend in the list. The public APIs handle nullptr here
  /// correctly, and the protected APIs (for the subclasses) should ignore the
  /// presence of this delegate entirely.
  LLCL::RCRef<BlobCacheBackend> delegate;
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
class BlobCache : public LLCL::ReferenceCounted<BlobCache<KeyInfo>> {
public:
  explicit BlobCache(LLCL::RCRef<BlobCacheBackend> backendList)
      : runtime(backendList->getRuntime()),
        backendList(std::move(backendList)) {
    // Only one additional type we need registered on top of the ones provided
    // by the backend list.
    LLCL::AsyncValue::registerTypes<ErrorOr<std::string>>();
  }

  using KeyTy = typename KeyInfo::KeyTy;

  LLCL::Runtime &getRuntime() { return runtime; }

  /// Store an item in the provided backends. On a collision, the backends are
  /// expected to overwrite the existing contents, so it is incumbent on the
  /// user to use a strong hash function! Returns the cache key on success -
  /// this can be used for speeding up future hash computations or simply
  /// discarded.
  LLCL::AsyncValueRef<ErrorOr<std::string>> insert(KeyTy key, BufferRef obj) {
    std::string keyHash = KeyInfo::hashKey(key);
    auto insertAsync =
        backendList->insert(Buffer::get(keyHash), std::move(obj));

    // Allocate a space for the output.
    auto out = LLCL::AsyncValueRef<ErrorOr<std::string>>::allocate(runtime);
    insertAsync.andThenSync([keyHash = std::move(keyHash), out = out.copy(),
                             insertAsync = insertAsync.copy()] {
      // If insertion failed, propagate the error. Otherwise, hand over the key
      // hash.
      if (insertAsync->isError())
        out.emplace(insertAsync->takeError());
      else
        out.emplace(keyHash);
    });

    return out;
  }

  /// Check if any of the provided backends have the item.
  LLCL::AsyncValueRef<bool> contains(KeyTy key) const {
    auto hash = Buffer::get(KeyInfo::hashKey(key));
    return backendList->contains(std::move(hash));
  }

  /// Get the item from any of the provided backends.
  LLCL::AsyncValueRef<CacheFindResult> find(KeyTy key) {
    auto hash = Buffer::get(KeyInfo::hashKey(key));
    return backendList->find(std::move(hash));
  }

  LLCL::AsyncValueRef<ErrorOrSuccess> clear() { return backendList->clear(); }

private:
  LLCL::Runtime &runtime;

  LLCL::RCRef<BlobCacheBackend> backendList;
};

/// Returns an in-memory implementation of the BlobCacheBackend.
LLCL::RCRef<BlobCacheBackend> getInMemoryBackend(LLCL::Runtime &runtime);

/// Returns a filesystem-based implementation of the BlobCacheBackend. If the
/// base path is not specified, then the backend will use the CWD.
LLCL::RCRef<BlobCacheBackend>
getFilesystemBackend(LLCL::Runtime &runtime,
                     const std::filesystem::path &basePath = "");

/// Returns a chain of pre-setup backends that represent the default chain,
/// inMemory->filesystem. The `basePath` is passed to getFilesystemBackend
/// directly.
ErrorOr<LLCL::RCRef<BlobCacheBackend>>
getDefaultBackendChain(LLCL::Runtime &runtime,
                       const std::filesystem::path &basePath = "");
} // namespace M::Cache

#endif // CACHE_BLOBCACHE_H
