//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Cache/Backends/HTTPCacheBackend.h"
#include "Support/Base64.h"
#include "llvm/Support/FormatVariadic.h"

using namespace M;
using namespace Cache;

ErrorOrSuccess HTTPCacheBackend::insertImpl(StringRef keyHash, BufferRef obj) {
  return Error::getStaticString("HTTP backend does not support insert");
}

ErrorOr<bool> HTTPCacheBackend::containsImpl(StringRef keyHash) const {
  auto findOr = findImpl(keyHash, std::nullopt);
  if (findOr.isError())
    return findOr.takeError();

  return findOr->has_value();
}

ErrorOr<std::optional<BufferRef>>
HTTPCacheBackend::findImpl(StringRef keyHash,
                           std::optional<WriteableBufferRef> buf) const {
  // First, check to see if we've already got this in our local cache.
  auto localBuf = const_cast<HTTPCacheBackend *>(this)->findBuffer(keyHash);
  if (localBuf) {
    // If we weren't passed a buffer, return the ref we have.
    if (!buf)
      return std::move(*localBuf);

    // If we were, then write the contents of our local buffer into the thing we
    // were given.
    (*buf)->write((*localBuf)->getBufferStart(), (*localBuf)->getBufferSize());
    return std::move(*buf);
  }

  // Base64 encode the key hash so it's URL-safe.
  std::string keyHashB64 = encodeURLSafeBase64(keyHash);

  // We didn't have it locally, so create a request to go get it.
  HTTPClient client(ctx.copy());
  HTTPRequest req{
      /*URL=*/url + "/" + keyHashB64,
      /*verifyTLSPeer=*/true,
  };

  // 10min timeout for requests.
  using namespace std::chrono_literals;
  constexpr std::chrono::milliseconds timeout = 10min;
  // Maximum of 512M per request.
  constexpr size_t maxBytes = 1024 * 1024 * 512;

  // Either create a new WriteableBuffer or use the one that was passed in.
  auto writeBuf =
      buf.has_value()
          ? std::move(*buf)
          : WriteableBuffer::get(/*size=*/0, /*alignment=*/std::nullopt,
                                 /*capacity=*/maxBytes);

  // Execute the request.
  HTTPResponse response =
      client.executeRequest(req, *writeBuf, timeout, maxBytes);

  // TODO: Will the result bytes be encoded or can we expect them to be raw
  //       binary?

  // Everything is fine, return the buffer.
  if (response.isSuccess()) {
    // Cache it, so we can avoid multiple requests at this level.
    const_cast<HTTPCacheBackend *>(this)->cacheBuffer(keyHash, writeBuf.copy());
    return std::move(writeBuf);
  }

  // Content was not found - this is not an error, just return nullopt.
  if (response.isError() && response.responseCode &&
      *response.responseCode == HTTPResponseCode::NotFound)
    return std::nullopt;

  // Return the error we hit.
  std::string errorContextStr =
      llvm::formatv("Looking for {0}", keyHashB64).str();
  return response.asError(errorContextStr).takeError();
}

ErrorOrSuccess HTTPCacheBackend::clearImpl() {
  return Error::getStaticString("HTTP backend does not support clear");
}

void HTTPCacheBackend::cacheBuffer(StringRef keyHash, BufferRef buf) {
  localCache.modify(
      [&](llvm::StringMap<BufferRef> &map) { map[keyHash] = std::move(buf); });
}

std::optional<BufferRef> HTTPCacheBackend::findBuffer(StringRef keyHash) {
  return localCache.read(
      [&](llvm::StringMap<BufferRef> &map) -> std::optional<BufferRef> {
        auto found = map.find(keyHash);
        if (found == map.end())
          return std::nullopt;
        return found->second.copy();
      });
}

HTTPCacheBackendRef M::Cache::getHTTPCacheBackend(HTTPContextRef ctx,
                                                  std::string url,
                                                  Runtime &runtime) {
  return HTTPCacheBackendRef::create(std::move(ctx), std::move(url), runtime);
}
