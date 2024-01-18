//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef CACHE_HTTPCACHEBACKEND_H
#define CACHE_HTTPCACHEBACKEND_H

#include "Cache/BlobCache.h"
#include "Support/HTTP/HTTPClient.h"
#include "Support/Progress.h"

/// Forward-declaration for the LLCL::Runtime.
namespace M::LLCL {
class Runtime;
}

namespace M::Cache {
/// This class provides a wrapper around libcurl that conforms to the
/// BlobCacheBackend interface. This enables the CAS to interact with an HTTP
/// upstream.
class HTTPCacheBackend : public BlobCacheBackend {
public:
  HTTPCacheBackend(HTTPContextRef c, std::string url, LLCL::Runtime &runtime,
                   Progress *progress)
      : BlobCacheBackend(runtime), ctx(std::move(c)), url(std::move(url)),
        progress(progress) {}

  /// Insert a binary blob with a PUT request.
  // TODO: Currently unsupported
  ErrorOrSuccess insertImpl(StringRef keyHash, BufferRef obj) override;

  /// Check if we have the object we're looking for. Since really the only way
  /// to check if we have the thing is to fetch it, we just get it and trust our
  /// in-memory request cache.
  ErrorOr<bool> containsImpl(StringRef keyHash) const override;

  /// Find the object at `keyHash`. This essentially produces a GET request to
  /// <url>/<urlsafe-b64-key-hash>, which is expected to return the bytes of the
  /// object directly.
  ErrorOr<std::optional<BufferRef>>
  findImpl(StringRef keyHash,
           std::optional<WriteableBufferRef> buf) const override;

  /// Underlying implementation for both containsImpl and findImpl. There is a
  /// need to distinguish the full request from simple the HEAD.
  ErrorOr<std::optional<BufferRef>>
  requestImpl(StringRef keyHash, std::optional<WriteableBufferRef> buf,
              bool headOnly) const;

  /// We do not support clear for this backend impl.
  ErrorOrSuccess clearImpl() override;

private:
  /// The HTTP context ref for any clients we spin up for a find operation.
  HTTPContextRef ctx;
  /// The base URL for requests.
  std::string url;
  /// Optional progress used for constructing requests.
  Progress *progress;

  /// Store a buffer in our local in-memory cache. We can't always rely on the
  /// backends ahead of us for this, because our `containsImpl` just runs
  /// `findImpl` manually. If we have a find first though, backends before us
  /// (notably the in-memory backend) will perform this same function.
  void cacheBuffer(StringRef keyHash, BufferRef buf);

  /// Check if we have a buffer in our local in-memory cache. If we do, return
  /// it.
  std::optional<BufferRef> findBuffer(StringRef keyHash);

  /// Local cache we can use to store buffers we've fetched from upstream. This
  /// is important because of the way we had to implement `containsImpl` - HTTP
  /// doesn't really provide a way to check if a resource exists without
  /// just...getting the resource.
  Shared<llvm::StringMap<BufferRef>> localCache;
};

using HTTPCacheBackendRef = RCRef<HTTPCacheBackend>;

/// Get the HTTP CAS backend. The backend will use `url` as the base, and append
/// the url-safe base-64 encoded key hash as the resource path.
HTTPCacheBackendRef getHTTPCacheBackend(HTTPContextRef ctx, std::string url,
                                        LLCL::Runtime &runtime,
                                        Progress *progress);
} // namespace M::Cache

#endif // CACHE_HTTPCACHEBACKEND_H
